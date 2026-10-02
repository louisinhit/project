"""Train the reference LANet jointly from random initialization."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.dont_write_bytecode = True
for key, value in {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_CACHE_DISABLE": "1",
    "MPLCONFIGDIR": str(ROOT / "_runtime" / "matplotlib"),
    "XDG_CACHE_HOME": str(ROOT / "_runtime" / "cache"),
    "TORCH_HOME": str(ROOT / "_runtime" / "torch"),
    "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "_runtime" / "inductor"),
    "TRITON_CACHE_DIR": str(ROOT / "_runtime" / "triton"),
    "TMPDIR": str(ROOT / "_runtime" / "tmp"),
}.items():
    os.environ[key] = value
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"], "torch", "kernels").mkdir(parents=True, exist_ok=True)
tempfile.tempdir = os.environ["TMPDIR"]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lanet_data import CSPBDataset
from lanet_model import CROP_LENGTHS, LANet, verify_reference_structure
from plot_loss import plot_run


REFERENCE_LOG = PROJECT / "logs_fusion" / "l_2022_n0.3_boi_1_lag_True_stage_3.log"
REFERENCE_CHECKPOINT = REFERENCE_LOG.with_suffix(".model")
STEP_FIELDS = ["epoch", "step_in_epoch", "global_step", "samples_seen", "interval_steps",
               "interval_samples", "loss_mean", "loss_last", "accuracy", "lr_used", "lr_next",
               "last_crop_length", "gradient_norm_last", "elapsed_seconds"] + [
                   f"batches_length_{length}" for length in CROP_LENGTHS]
EPOCH_FIELDS = ["epoch", "global_step", "train_samples", "train_loss", "train_accuracy",
                "monitor_samples", "monitor_loss", "monitor_accuracy", "monitor_length",
                "lr_next", "epoch_seconds", "elapsed_seconds"] + [
                    f"batches_length_{length}" for length in CROP_LENGTHS]


def inside_test(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Refusing to write outside test/: {path}")
    return path


def atomic_json(path, value):
    path = inside_test(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = inside_test(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LossAccumulator:
    def __init__(self):
        self.samples = self.steps = self.correct = 0
        self.loss_sum = 0.0
        self.length_batches = dict.fromkeys(CROP_LENGTHS, 0)

    def add(self, loss, count, correct, length):
        self.loss_sum += loss * count
        self.samples += count
        self.correct += correct
        self.steps += 1
        self.length_batches[length] += 1

    @property
    def loss(self):
        return self.loss_sum / self.samples

    @property
    def accuracy(self):
        return self.correct / self.samples

    def length_fields(self):
        return {f"batches_length_{length}": count for length, count in self.length_batches.items()}


def open_csv(path, fields, resume=False):
    stream = path.open("a" if resume else "x", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    if not resume:
        writer.writeheader()
        stream.flush()
    return stream, writer


def trim_csv(path, fields, key, maximum):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row[key]) <= maximum]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    accumulator = LossAccumulator()
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite held-out loss.")
        accumulator.add(loss.item(), target.numel(), (output.argmax(1) == target).sum().item(), 32768)
    return accumulator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a test/ run from its last completed epoch.")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT.parent / "dataset" / "CSPB.ML" / "CSPB_ML_2022_Data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dense-epochs", type=int, default=10)
    parser.add_argument("--dense-interval", type=int, default=10)
    parser.add_argument("--sparse-interval", type=int, default=100)
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--noise-max", type=float, default=0.3)
    parser.add_argument("--iq-standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke tests only; 0 uses all 80000.")
    parser.add_argument("--monitor-limit", type=int, default=0, help="Smoke tests only; 0 uses all 32000.")
    args = parser.parse_args()
    resume = inside_test(args.resume) if args.resume else None
    if resume:
        with (resume / "config.json").open() as stream:
            saved = json.load(stream)["arguments"]
        args = argparse.Namespace(**saved)
        args.run_dir, args.data_root, args.resume = resume, Path(args.data_root), resume
    elif args.run_dir is None:
        args.run_dir = ROOT / "losses" / datetime.now().strftime("lanet_joint_%Y%m%d_%H%M%S")
    args.run_dir = inside_test(args.run_dir)
    for field in ("epochs", "batch_size", "threads", "dense_interval", "sparse_interval"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.workers < 0 or args.dense_epochs < 0:
        parser.error("Batch size must be >= 2; workers and dense epochs must be >= 0.")
    if args.base_lr <= 0 or not math.isfinite(args.base_lr) or not 0 <= args.noise_max <= 1:
        parser.error("Invalid learning rate or noise amplitude.")
    return args


def main():
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and (run_dir / "config.json").exists():
        raise FileExistsError("Run already exists; use --resume or choose another run directory.")
    status = {"state": "initializing", "pid": os.getpid(), "epoch": 0, "global_step": 0,
              "planned_epochs": args.epochs, "run_dir": str(run_dir)}
    atomic_json(run_dir / "status.json", status)
    streams = []
    started = time.monotonic()
    elapsed_offset = 0.0
    try:
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no silent CPU fallback.")
        model = LANet(args.noise_max, args.iq_standardize)
        verification = verify_reference_structure(model, REFERENCE_CHECKPOINT)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr,
                                      betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.base_lr, max_lr=args.base_lr * 5,
            step_size_up=1600, mode="triangular", cycle_momentum=False,
        )
        source_paths = [ROOT / name for name in ("train_lanet.py", "lanet_model.py", "lanet_data.py", "plot_loss.py")]
        source_paths += [PROJECT / "code_lag" / name for name in ("model_wrapper.py", "model_mlp.py")]
        hashes = {str(path): sha256(path) for path in source_paths}
        config = {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_log": str(REFERENCE_LOG), "reference_log_sha256": sha256(REFERENCE_LOG),
            "reference_checkpoint": str(REFERENCE_CHECKPOINT),
            "reference_checkpoint_sha256": sha256(REFERENCE_CHECKPOINT),
            "structure_verification": verification, "source_sha256": hashes,
            "crop_lengths": list(CROP_LENGTHS), "train_batches": list(range(1, 21)),
            "monitor_batches": list(range(21, 29)), "monitor_length": 32768,
            "truth_labels_sha256": sha256(args.data_root / "CSPB_ML_2022_Signal_Truth_Labels.txt"),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "python_version": sys.version, "numpy_version": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "numeric_settings": {"amp": False, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                 "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                 "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
            "provenance_note": "New joint-from-scratch run, not the original staged-training history. "
                               "Architecture verified by log and state tensor shapes; parameter-free historical "
                               "forward operations are not fully recoverable. IQ standardization follows current code_lag.",
        }
        if not args.resume:
            atomic_json(run_dir / "config.json", config)
            (run_dir / "model.txt").write_text(str(model) + "\n")
        else:
            with (run_dir / "config.json").open() as stream:
                original_config = json.load(stream)
            for key in ("source_sha256", "truth_labels_sha256", "reference_checkpoint_sha256", "reference_log_sha256"):
                if config[key] != original_config[key]:
                    raise ValueError(f"Resume provenance changed: {key}")
        print(f"Run: {run_dir}\nStructure verified: {verification}\nDevice: {device}", flush=True)
        status["state"] = "loading_data"
        atomic_json(run_dir / "status.json", status)
        trainset = CSPBDataset(args.data_root, range(1, 21), args.train_limit)
        monitorset = CSPBDataset(args.data_root, range(21, 29), args.monitor_limit)
        if len(trainset) % args.batch_size == 1 or len(monitorset) % args.batch_size == 1:
            raise ValueError("Choose a batch size that does not leave a singleton batch (original BOI helper).")
        print(f"RAM datasets: train={len(trainset)}, held-out={len(monitorset)}", flush=True)
        generator = torch.Generator()
        loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
        train_loader = DataLoader(trainset, shuffle=True, generator=generator, **loader_options)
        monitor_loader = DataLoader(monitorset, shuffle=False,
                                    generator=torch.Generator().manual_seed(args.seed + 100000), **loader_options)
        for name in ("loss", "checkpoints", "plots"):
            (run_dir / name).mkdir(exist_ok=True)
        first_epoch, global_step, samples_seen = 1, 0, 0
        best_monitor_loss = float("inf")
        if args.resume:
            checkpoint = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            first_epoch = checkpoint["epoch"] + 1
            global_step, samples_seen = checkpoint["global_step"], checkpoint["samples_seen"]
            best_monitor_loss = checkpoint["best_monitor_loss"]
            elapsed_offset = checkpoint["elapsed_seconds"]
            restore_rng(checkpoint["rng_state"])
            trim_csv(run_dir / "loss" / "steps.csv", STEP_FIELDS, "global_step", global_step)
            trim_csv(run_dir / "loss" / "epochs.csv", EPOCH_FIELDS, "epoch", first_epoch - 1)
            print(f"Resuming at epoch {first_epoch}, step {global_step}", flush=True)
        step_stream, step_writer = open_csv(run_dir / "loss" / "steps.csv", STEP_FIELDS, bool(args.resume))
        epoch_stream, epoch_writer = open_csv(run_dir / "loss" / "epochs.csv", EPOCH_FIELDS, bool(args.resume))
        streams = [step_stream, epoch_stream]
        for epoch in range(first_epoch, args.epochs + 1):
            epoch_started = time.monotonic()
            generator.manual_seed(args.seed + epoch)
            model.train()
            totals, window = LossAccumulator(), LossAccumulator()
            interval = args.dense_interval if epoch <= args.dense_epochs else args.sparse_interval
            status.update(state="training", epoch=epoch, global_step=global_step)
            atomic_json(run_dir / "status.json", status)
            for batch_step, (data, target) in enumerate(train_loader, 1):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                lr_used = optimizer.param_groups[0]["lr"]
                output = model(data)
                loss = F.nll_loss(output, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_step}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                global_step += 1
                count = target.numel()
                samples_seen += count
                loss_value = loss.item()
                correct = (output.detach().argmax(1) == target).sum().item()
                length = model.preprocess.last_crop_length
                totals.add(loss_value, count, correct, length)
                window.add(loss_value, count, correct, length)
                if window.steps == interval or batch_step == len(train_loader):
                    elapsed = elapsed_offset + time.monotonic() - started
                    step_writer.writerow({
                        "epoch": epoch, "step_in_epoch": batch_step, "global_step": global_step,
                        "samples_seen": samples_seen, "interval_steps": window.steps,
                        "interval_samples": window.samples, "loss_mean": window.loss,
                        "loss_last": loss_value, "accuracy": window.accuracy, "lr_used": lr_used,
                        "lr_next": optimizer.param_groups[0]["lr"], "last_crop_length": length,
                        "gradient_norm_last": gradient_norm.item(), "elapsed_seconds": elapsed,
                        **window.length_fields(),
                    })
                    step_stream.flush()
                    status.update(global_step=global_step, step_in_epoch=batch_step, loss=window.loss,
                                  elapsed_seconds=elapsed, samples_seen=samples_seen)
                    atomic_json(run_dir / "status.json", status)
                    print(f"Epoch {epoch}/{args.epochs} step {batch_step}/{len(train_loader)} "
                          f"global={global_step} loss={window.loss:.6f} "
                          f"lr={optimizer.param_groups[0]['lr']:.7f} L={length}", flush=True)
                    window = LossAccumulator()
            status["state"] = "evaluating"
            atomic_json(run_dir / "status.json", status)
            monitor = evaluate(model, monitor_loader, device)
            elapsed = elapsed_offset + time.monotonic() - started
            epoch_writer.writerow({
                "epoch": epoch, "global_step": global_step, "train_samples": totals.samples,
                "train_loss": totals.loss, "train_accuracy": totals.accuracy,
                "monitor_samples": monitor.samples, "monitor_loss": monitor.loss,
                "monitor_accuracy": monitor.accuracy, "monitor_length": 32768,
                "lr_next": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": elapsed,
                **totals.length_fields(),
            })
            for stream in streams:
                stream.flush()
                os.fsync(stream.fileno())
            improved = monitor.loss < best_monitor_loss
            best_monitor_loss = min(best_monitor_loss, monitor.loss)
            checkpoint = {
                "epoch": epoch, "global_step": global_step, "samples_seen": samples_seen,
                "best_monitor_loss": best_monitor_loss, "elapsed_seconds": elapsed,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "rng_state": capture_rng(),
            }
            atomic_checkpoint(run_dir / "checkpoints" / "last.pt", checkpoint)
            if improved:
                atomic_checkpoint(run_dir / "checkpoints" / "best.pt", {
                    "model_state_dict": model.state_dict(), "epoch": epoch,
                    "monitor_loss": monitor.loss, "monitor_accuracy": monitor.accuracy,
                })
            plot_run(run_dir)
            print(f"Epoch {epoch} complete: train_loss={totals.loss:.6f}, "
                  f"heldout_loss={monitor.loss:.6f}, heldout_acc={monitor.accuracy:.4%}", flush=True)
            status.update(state="epoch_complete", completed_epochs=epoch,
                          train_loss=totals.loss, monitor_loss=monitor.loss,
                          monitor_accuracy=monitor.accuracy, best_monitor_loss=best_monitor_loss,
                          elapsed_seconds=elapsed)
            atomic_json(run_dir / "status.json", status)
        status.update(state="completed", completed_epochs=args.epochs, global_step=global_step,
                      elapsed_seconds=elapsed_offset + time.monotonic() - started)
        atomic_json(run_dir / "status.json", status)
        print("Training completed.", flush=True)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        atomic_json(run_dir / "status.json", status)
        traceback.print_exc()
        raise
    finally:
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == "__main__":
    main()
