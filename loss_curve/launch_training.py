"""Start training detached; every generated file stays under test/."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    args, extra = parser.parse_known_args()
    if args.resume:
        run_dir = args.resume.resolve()
        arguments = ["--resume", str(run_dir)]
    else:
        run_dir = ROOT / "losses" / datetime.now().strftime("lanet_joint_%Y%m%d_%H%M%S")
        arguments = ["--run-dir", str(run_dir), *extra]
    if not run_dir.is_relative_to(ROOT):
        parser.error("The run must remain inside test/.")
    run_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", CUDA_CACHE_DISABLE="1")
    with (run_dir / "console.log").open("a" if args.resume else "x") as stream:
        process = subprocess.Popen(
            [sys.executable, "-B", "-u", str(ROOT / "train_lanet.py"), *arguments],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    (run_dir / "pid.txt").write_text(f"{process.pid}\n")
    print(json.dumps({"pid": process.pid, "run_dir": str(run_dir),
                      "console": str(run_dir / "console.log")}, indent=2))


if __name__ == "__main__":
    main()
