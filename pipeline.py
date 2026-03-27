"""
pipeline.py — Single entry point for the full FMCG supply-chain pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW TO PROVIDE YOUR INPUT CSV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Option 1 — Pass it directly (most common):
      python pipeline.py --input sales_data.csv
      python pipeline.py --input C:/data/march_2026_sales.csv

  Option 2 — Drop it in the inputs/ folder:
      Place your .csv file in the inputs/ folder next to pipeline.py.
      Then just run:
          python pipeline.py
      The pipeline will detect and use it automatically.

  Option 3 — Use the hardcoded default (set in config.py):
      If no --input flag and no file in inputs/, falls back to
      INPUT_CSV_DEFAULT in config.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OTHER FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python pipeline.py --from stage2     # skip XGBoost, load checkpoint
  python pipeline.py --from stage3     # skip both, load inventory checkpoint
  python pipeline.py --no-checkpoints  # run without saving intermediate files
"""

import argparse
import glob
import os
import sys
import time

import pandas as pd

import config
import stage1_xgboost   as s1
import stage2_inventory as s2
import stage3_routes    as s3


# ── Input CSV resolution ───────────────────────────────────────────────────

def resolve_input_csv(cli_path) -> str:
    """
    Determine which CSV file to use, in priority order:
      1. --input flag from command line
      2. Any single .csv file dropped in the inputs/ folder
      3. INPUT_CSV_DEFAULT from config.py

    Returns the resolved absolute path, or exits with a helpful message.
    """

    # ── 1. Explicit CLI path ──────────────────────────────────────────────
    if cli_path:
        path = os.path.abspath(cli_path)
        if not os.path.exists(path):
            sys.exit(
                f"\n❌ File not found: {path}\n"
                f"   Double-check the path and try again.\n"
            )
        if not path.lower().endswith(".csv"):
            sys.exit(f"\n❌ Expected a .csv file, got: {path}\n")
        print(f"   📄 Input CSV (--input flag)   : {path}")
        return path

    # ── 2. Drop-zone folder ───────────────────────────────────────────────
    inputs_dir = config.INPUTS_DIR
    os.makedirs(inputs_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(inputs_dir, "*.csv"))

    if len(csv_files) == 1:
        path = os.path.abspath(csv_files[0])
        print(f"   📄 Input CSV (inputs/ folder) : {path}")
        return path

    if len(csv_files) > 1:
        names = "\n      ".join(os.path.basename(f) for f in csv_files)
        sys.exit(
            f"\n❌ Multiple CSV files found in inputs/ folder:\n"
            f"      {names}\n\n"
            f"   Please specify which one:\n"
            f"      python pipeline.py --input inputs/<filename>.csv\n"
        )

    # ── 3. Hardcoded default ──────────────────────────────────────────────
    path = os.path.abspath(config.INPUT_CSV_DEFAULT)
    if not os.path.exists(path):
        sys.exit(
            f"\n❌ No input CSV found. Tried:\n"
            f"   • --input flag        : not provided\n"
            f"   • inputs/ folder      : empty  ({inputs_dir})\n"
            f"   • config default path : {path}  (does not exist)\n\n"
            f"   Quickest fix:\n"
            f"      python pipeline.py --input your_sales_data.csv\n"
        )
    print(f"   📄 Input CSV (config default) : {path}")
    return path


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _banner(title: str):
    print("\n" + "█" * 70)
    print(f"  {title}")
    print("█" * 70)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


# ── Stage runners ──────────────────────────────────────────────────────────

def run_stage1(csv_path: str, save_checkpoint: bool) -> pd.DataFrame:
    t   = time.time()
    raw = pd.read_csv(csv_path)
    print(f"   Rows loaded : {len(raw):,}  |  Columns : {list(raw.columns)}")

    matrix = s1.run(raw)

    if save_checkpoint and config.CHECKPOINT_XGBOOST:
        s1.save_checkpoint(matrix, config.CHECKPOINT_XGBOOST)

    print(f"   ⏱  Stage 1 took {_elapsed(t)}")
    return matrix


def run_stage2(matrix: pd.DataFrame, save_checkpoint: bool) -> pd.DataFrame:
    t         = time.time()
    transfers = s2.run(matrix)

    if save_checkpoint and config.CHECKPOINT_INVENTORY:
        s2.save_checkpoint(transfers, config.CHECKPOINT_INVENTORY)

    print(f"   ⏱  Stage 2 took {_elapsed(t)}")
    return transfers


def run_stage3(transfers: pd.DataFrame) -> pd.DataFrame:
    t = time.time()
    routes_df, report = s3.run(transfers)
    s3.save_checkpoint(routes_df, report, config.FINAL_OUTPUT)
    print(f"   ⏱  Stage 3 took {_elapsed(t)}")
    return routes_df


# ── Load from checkpoint ───────────────────────────────────────────────────

def _load_xgboost_checkpoint() -> pd.DataFrame:
    path = config.CHECKPOINT_XGBOOST
    if not path or not os.path.exists(path):
        sys.exit(
            f"\n❌ XGBoost checkpoint not found: {path}\n"
            f"   Run the full pipeline first:\n"
            f"      python pipeline.py --input your_data.csv\n"
        )
    print(f"   📂 Loading XGBoost checkpoint  : {path}")
    return pd.read_excel(path)


def _load_inventory_checkpoint() -> pd.DataFrame:
    path = config.CHECKPOINT_INVENTORY
    if not path or not os.path.exists(path):
        sys.exit(
            f"\n❌ Inventory checkpoint not found: {path}\n"
            f"   Run from stage2 or the full pipeline first:\n"
            f"      python pipeline.py --input your_data.csv\n"
        )
    print(f"   📂 Loading inventory checkpoint: {path}")
    return pd.read_excel(path, sheet_name="Detailed_Transfers")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FMCG Supply-Chain Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        metavar="CSV_PATH",
        default=None,
        help="Path to sales input CSV file (see options above if not provided)",
    )
    parser.add_argument(
        "--from",
        dest="start_stage",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
        help="Start from a specific stage using saved checkpoints",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Run without saving intermediate checkpoint files",
    )
    args = parser.parse_args()

    save_checkpoints = not args.no_checkpoints
    start            = args.start_stage

    _ensure_output_dir()
    total_start = time.time()

    _banner("FMCG SUPPLY-CHAIN PIPELINE")
    print(f"   Start stage  : {start}")
    print(f"   Checkpoints  : {'enabled' if save_checkpoints else 'disabled'}")
    print(f"   Output       : {config.FINAL_OUTPUT}")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if start == "stage1":
        csv_path = resolve_input_csv(args.input)
        matrix   = run_stage1(csv_path, save_checkpoints)
    else:
        # CSV not needed when resuming from stage 2 or 3
        matrix = _load_xgboost_checkpoint()

    # ── Stage 2 ──────────────────────────────────────────────────────────
    if start in ("stage1", "stage2"):
        transfers = run_stage2(matrix, save_checkpoints)
    else:
        transfers = _load_inventory_checkpoint()

    # ── Stage 3 ──────────────────────────────────────────────────────────
    routes_df = run_stage3(transfers)

    # ── Done ──────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"   Total time   : {_elapsed(total_start)}")
    print(f"   Final output : {config.FINAL_OUTPUT}")
    print(f"   Routes       : {len(routes_df)}")
    print(f"\n   Open outputs/Route.xlsx to review the delivery plan.\n")


if __name__ == "__main__":
    main()
