"""Windows-compatible one-shot pipeline runner.

Replaces `make all` on systems without GNU Make (e.g., Windows).
Usage: python run_all.py [--output PREFIX]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 mode for child processes on Windows before any heavy imports.
os.environ.setdefault("PYTHONUTF8", "1")


def run(cmd: list[str], cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {' '.join(cmd)}")
    print("=" * 60)
    # cmd is a list; no shell=True — avoids shell-injection surface and
    # correctly handles paths with spaces.
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Marketing Attribution full pipeline runner")
    parser.add_argument("--output", type=str, default=None, help="Output path prefix for generated files")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # Materialize output dirs explicitly (config import is now side-effect-free).
    from attributor.config import ensure_dirs

    ensure_dirs()

    # Build step commands as argv lists; append --output if provided.
    preprocess_cmd = ["python", "-m", "attributor.preprocess"]
    if args.output:
        preprocess_cmd += ["--output", args.output]

    # NOTE on the attribution path: `make all` (Makefile) runs preprocess_criteo
    # (REAL Criteo journeys), while this runner historically ran generate_touchpoints
    # (SYNTHETIC). To unify the two, we attempt the real Criteo preprocessing and
    # only fall back to synthetic generation if the raw Criteo TSV is absent (it is
    # ~16M rows and not bundled). multi_touch_attribution.load_data() itself also
    # falls back from Criteo to simulated parquet, so the downstream step is robust
    # either way.
    def _has_criteo():
        from attributor.config import CRITEO_RAW_PATH  # local import keeps top level cheap

        return CRITEO_RAW_PATH.exists()

    if _has_criteo():
        attribution_prep = ("Criteo preprocessing", ["python", "-m", "attributor.preprocess_criteo"])
    else:
        attribution_prep = ("Touchpoint Generation", ["python", "-m", "attributor.generate_touchpoints"])

    steps = [
        ("Preprocessing", preprocess_cmd),
        ("MMM Modeling", ["python", "-m", "attributor.mmm_model"]),
        attribution_prep,
        ("Multi-touch Attribution", ["python", "-m", "attributor.multi_touch_attribution"]),
        ("Budget Optimization", ["python", "-m", "attributor.budget_optimizer"]),
    ]

    print("Marketing Attribution & Budget Optimization - Full Pipeline")
    print("=" * 60)

    for name, cmd in steps:
        if not run(cmd, cwd=here):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
