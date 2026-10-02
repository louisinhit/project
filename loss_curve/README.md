# LANet Joint Retraining and Loss Curves

## Started Experiment

The initial full 50-epoch run was launched on 2026-10-02:

```text
Run: /home/lxy/ab_vary_length/test/losses/lanet_joint_20261002_031341
Initial training PID: 438367
```

Read that run's `status.json` for live progress and `console.log` for errors.
The initial PID may change after a restart; `pid.txt` is maintained by the
launcher. The configured IQ standardization assumption is described below.

## Purpose and Scope

This is a **new end-to-end training experiment from random initialization**,
requested for a thesis correction. It does not recover the historical loss
curve of the original IQ/CC pretraining, frozen-fusion training and fine-tuning.
Do not describe its curves as the history of the original staged experiment,
or attribute the original reported accuracy to this new run without evaluation.

All scripts and generated artifacts are under `/home/lxy/ab_vary_length/test`.
Existing model code, datasets, logs and checkpoints are read only. Python
bytecode generation and CUDA disk caching are disabled; temporary files and
Matplotlib caches are redirected to `test/_runtime/`. No dependencies are
installed, and no disk copy of the dataset is created.

## Reference Model and Reconstruction

Reference:

```text
../logs_fusion/l_2022_n0.3_boi_1_lag_True_stage_3.log
../logs_fusion/l_2022_n0.3_boi_1_lag_True_stage_3.model
```

The current `code_lag` model is **not** the architecture in this log. Its IQ
backbone and fusion head have changed. `lanet_model.py` reconstructs the logged
architecture and verifies every state tensor name/shape against the checkpoint.
This check never calls `load_state_dict` with the reference weights. All
3,238,312 parameters are initialized afresh and remain trainable together.

| Component | Configuration |
| --- | --- |
| IQ / CC inputs | 2 / 8 real channels |
| Both backbones | Conv widths 32, 48, 64, 96, 128; kernel 23; same padding |
| Downsampling | MaxPool1d(2) after each of five conv blocks |
| Final temporal conv | 128 to 256; kernel 23; same padding |
| Statistics pooling | Concatenated temporal mean and population standard deviation |
| Embedding per branch | Linear(512,256), BatchNorm1d(256), ReLU |
| LAG projections | Linear(256,256), Tanh, Dropout(0.1), per branch |
| Gate | Linear(513,256), Sigmoid; concatenated IQ, CC and log2(L)/log2(32768) |
| Fusion | gate * projected_CC + (1 - gate) * projected_IQ |
| Classifier | Linear(256,512), GELU, Linear(512,8) |

The log and state dictionary cannot establish all parameter-free historical
forward operations. In particular, whether the old IQ path used per-channel
standardization cannot be recovered with certainty. The default follows the
current `code_lag/model_wrapper.py`: standardize IQ and CC along time with
sample standard deviation and epsilon 1e-5. The related `aa_vary_length` IQ
path does not do this. `--no-iq-standardize` is available if the historical
choice is confirmed to be different. The choice is saved in `config.json`.
No claim of exact historical forward-code recovery is made.

The existing BOI/normalization, CC extraction, convolution blocks and statistics
pooling helpers are reused read only. Their source hashes are recorded, and
resume rejects changes to those sources. CC channels preserve their existing
order: abs(x^2), abs(FFT(x^2)), abs(x^4), abs(FFT(x^4)), abs(x^8),
abs(FFT(x^8)), abs(x^6), abs(FFT(x^6)); FFT uses orthonormal scaling and fftshift.

## Data and Training Protocol

| Setting | Default |
| --- | --- |
| Dataset | CSPB.ML.2022, full SNR range |
| Training split | Batch_Dir_1 through Batch_Dir_20: 80,000 signals |
| Held-out monitoring split | Batch_Dir_21 through Batch_Dir_28: 32,000 signals |
| Classes, in order | bpsk, qpsk, 8psk, dqpsk, msk, 16qam, 64qam, 256qam |
| Epochs / batch size | 50 / 160: 500 steps per epoch, 25,000 optimizer steps |
| Seed | 1234 for Python, NumPy and PyTorch |
| Noise | Original complex torch.randn_like, batch-level amplitude uniform in [0,0.3) |
| Training lengths | 2048, 4096, 8192, 16384, 32768; uniform choice per minibatch |
| Crop start | Original randint(0,16380), clamped to 32768 - L |
| BOI | Enabled; threshold ratio 14; original width-15 smoothing and centering |
| Signal normalization | Original per-signal unit average power after BOI |
| Loss | Mean negative log likelihood of log-softmax outputs, equivalent to cross-entropy of logits |
| Optimizer | AdamW, betas=(0.9,0.999), eps=1e-6, weight_decay=0 |
| Learning rate | CyclicLR triangular: 5e-4 to 2.5e-3; 1600 steps up and down |
| Gradient clipping | Global norm 1.0 |
| Precision | float32 real network / complex64 signal; no AMP |
| Monitoring | Each epoch, full-length 32768 signals, no added noise, no random crop |

The learning-rate settings follow **stage 0 (end-to-end)** in the original
`run_tasks.py`, not the stage-3 fine-tuning rate of 1e-5. This is an intentional
difference appropriate to fresh initialization. Length candidates and the
50-epoch budget are also intentional changes. There is no branch pretraining,
parameter freezing, auxiliary branch loss or loading of previous model weights.

The monitoring split was named "test 2022" in the original code. It has not
been silently repartitioned into a new validation set, and must not be claimed
as an untouched independent test set after it is used for checkpoint selection.
The 2018 cross-dataset evaluation is not run here. Train and monitoring losses
have different length/noise distributions, so their absolute values are not
directly comparable measures of overfitting.

The reader validates the truth-label class order against the original cyclic
labels and preserves the original float16 BOI bounds before casting to
complex64. Signals and metadata are preallocated in RAM, approximately
27.3 GiB total for both splits. No samples or final incomplete batches are
dropped. A singleton final minibatch is rejected because the reused original
BOI helper cannot handle it. The default split/batch sizes have no remainder.

Seeds, sample order and RNG state are saved/reproducible. cuDNN benchmarking is
disabled; other PyTorch numerical defaults are recorded rather than silently
changed. Bitwise determinism across hardware/versions is **not guaranteed**.

## Files and Output Layout

```text
test/
  README.md
  lanet_model.py          reconstructed architecture; read-only reuse of helpers
  lanet_data.py           original split and signal decoding; RAM-only data
  train_lanet.py          training, monitoring, CSVs, checkpoints and resume
  launch_training.py     detached launcher with PID and console log
  plot_loss.py            regenerate PNG/PDF plots from CSVs
  test_workflow.py        focused verification
  _runtime/              local caches and temporary runtime files
  losses/
    lanet_joint_YYYYMMDD_HHMMSS/
      config.json        settings, provenance hashes, device and versions
      model.txt          model structure
      status.json        state, progress and last measured metrics
      pid.txt            detached training PID
      console.log        training stdout/stderr
      loss/
        steps.csv        recorded minibatch-window loss measurements
        epochs.csv       complete epoch averages and held-out metrics
      checkpoints/
        last.pt          most recent completed epoch, optimizer, scheduler, RNG
        best.pt          weights with lowest held-out 2022 monitoring loss
      plots/
        training_steps.png / .pdf
        epoch_loss.png / .pdf
```

Smoke-test outputs belong in `test/verification_runs/`, not alongside the full
experiment. Only two checkpoints are retained; they are replaced atomically.
Typical persistent checkpoint storage is about 50 MiB, plus small CSVs/plots.
Atomic replacement temporarily needs roughly one additional checkpoint's
space. No per-step/per-epoch checkpoint archive or feature/gate tensor dump is
created. Full-run CSVs contain about 700 step rows and 50 epoch rows.

### Loss Definitions

Epochs 1-10: record every **10 optimizer steps**. Epochs 11-50: every **100
steps**. Always record the final partial interval of an epoch.

`steps.csv` stores sample-weighted `loss_mean` across the complete recording
window, plus the last minibatch's `loss_last`. These are distinct quantities:
the plotted curve is the measured window mean, not a single-step sample or an
interpolated reconstruction. Every minibatch contributes to the saved window
mean and to the epoch mean, including steps between CSV writes.

Useful fields: epoch, step_in_epoch, global_step, samples_seen, interval_steps,
interval_samples, loss_mean, loss_last, accuracy, lr_used, lr_next,
last_crop_length, gradient_norm_last and elapsed_seconds. Five
`batches_length_*` columns record the length mixture in each interval.
`gradient_norm_last` is the norm **before** clipping.

`epochs.csv` stores sample-weighted train_loss/train_accuracy and
monitor_loss/monitor_accuracy, counts, learning rate, elapsed time and crop
counts. Accuracy is a fraction in [0,1]. Means are computed from actual sample
counts, without the original logger's off-by-one interval accumulation.

Plots are refreshed after every completed epoch and can also be generated
while training is running. The first-10-epoch panel uses only densely recorded
observations. A line connects existing points; no smoothing is applied.

## Commands on This Server

Use the existing `amc` environment; no installation is needed:

```bash
cd /home/lxy/ab_vary_length/test
PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 /home/lxy/miniconda3/envs/amc/bin/python -B test_workflow.py
```

Small real-data smoke run (not thesis results):

```bash
/home/lxy/miniconda3/envs/amc/bin/python -B -u train_lanet.py \
  --run-dir /home/lxy/ab_vary_length/test/verification_runs/smoke \
  --epochs 2 --batch-size 8 --workers 0 \
  --train-limit 32 --monitor-limit 16 \
  --dense-epochs 1 --dense-interval 1 --sparse-interval 3
```

Full run, detached from the terminal:

```bash
/home/lxy/miniconda3/envs/amc/bin/python -B launch_training.py
```

The launcher prints the actual run directory and PID. Use that path below;
`RUN` is an example placeholder, not another directory to create.

```bash
RUN=/home/lxy/ab_vary_length/test/losses/lanet_joint_YYYYMMDD_HHMMSS
cat "$RUN/status.json"
tail -n 20 "$RUN/console.log"
```

Regenerate plots without training:

```bash
/home/lxy/miniconda3/envs/amc/bin/python -B plot_loss.py "$RUN"
```

Resume after an interruption from the latest **completed epoch**:

```bash
/home/lxy/miniconda3/envs/amc/bin/python -B launch_training.py --resume "$RUN"
```

Resume restores the saved arguments, model, AdamW, CyclicLR, and CPU/CUDA/Python/
NumPy RNG states. Per-epoch shuffling uses seed + epoch. Uncheckpointed rows
after the restored epoch are removed from these experimental CSVs and that
epoch is rerun, avoiding duplicate progress. Sources/provenance must match.
Do not resume while the original process is still running; a process-held lock
rejects simultaneous writers. The run is not complete until `status.json`
reports `state: completed` and `completed_epochs: 50`.

For a deliberate stop, first inspect the PID in `pid.txt`, then send SIGINT to
that training PID only. Partial-epoch CSV rows are flushed, but the resumable
checkpoint remains the previous completed epoch. Nothing automatically kills
other GPU processes or deletes old runs.

Read numerical results directly with standard `csv`, NumPy or pandas. For
example, `pd.read_csv(f"{run}/loss/steps.csv")` gives the exact measurements
used by the step plots. Keep `config.json` with the CSVs when using figures in
the thesis response so that the altered training protocol is traceable.
