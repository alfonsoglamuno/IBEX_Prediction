"""
IBEX35 Daily Morning Run
========================
Run this script every morning before the trading session opens (e.g. 08:30 CET).
It refreshes market data, generates the prediction with SHAP, and saves the
operative summary so the dashboard is ready to use immediately.

Usage:
    python morning_run.py                   # standard run
    python morning_run.py --retrain         # retrain models first (weekly/monthly)
    python morning_run.py --no-shap         # skip SHAP (faster, ~10s vs ~40s)
    python morning_run.py --open-dashboard  # also launch the Streamlit dashboard

Windows Task Scheduler setup (runs every weekday at 08:30 CET):
    Program:  C:\\path\\to\\python.exe
    Arguments: C:\\path\\to\\IBEX_Prediction\\morning_run.py
    Start in: C:\\path\\to\\IBEX_Prediction

    Or use Task Scheduler wizard:
        schtasks /create /tn "IBEX_Morning_Run" /tr "python morning_run.py" ^
                 /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 08:30 ^
                 /sd 01/01/2026 /f
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make src importable when running from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import root
from src.utils.logging import get_logger

log = get_logger("morning_run")


def parse_args():
    p = argparse.ArgumentParser(description="IBEX35 daily morning run")
    p.add_argument("--retrain",          action="store_true",
                   help="Retrain all models before predicting (slow, for weekly use)")
    p.add_argument("--no-shap",          action="store_true",
                   help="Skip SHAP analysis (faster run)")
    p.add_argument("--open-dashboard",   action="store_true",
                   help="Launch Streamlit dashboard after generating predictions")
    p.add_argument("--no-summary",       action="store_true",
                   help="Skip generating the operative summary")
    return p.parse_args()


def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess and stream its output. Returns exit code."""
    log.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if proc.returncode == 0:
        log.info(f"{label} completed in {elapsed:.0f}s")
    else:
        log.error(f"{label} FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
    return proc.returncode


def _print_signal_banner(pred_path: Path) -> None:
    """Print a clean morning briefing from latest_prediction.json."""
    if not pred_path.exists():
        log.warning("No prediction file found.")
        return

    with open(pred_path) as f:
        preds = json.load(f)

    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    print("\n" + "=" * 60)
    print(f"  IBEX35 MORNING BRIEFING — {now}")
    print("=" * 60)

    for target, pred in preds.items():
        if "error" in pred:
            print(f"  {target}: ERROR — {pred['error']}")
            continue
        horizon = "1 day " if "1d" in target else "5 days"
        signal  = pred.get("signal", "?")
        prob_up = pred.get("prob_up", 0.5)
        conf    = pred.get("confidence", "?")
        model   = pred.get("model", "?")
        date    = pred.get("date", "?")

        color = {"UP": "↑", "DOWN": "↓", "NEUTRAL": "→"}.get(signal, "?")
        print(f"\n  {horizon:8s}  {color} {signal:7s}  P(up)={prob_up:.1%}  [{conf}]")
        print(f"           Model: {model}  |  As of: {date}")

        # Top 3 SHAP drivers
        drivers = pred.get("top_drivers", [])
        if drivers:
            print("           Top drivers:")
            for d in drivers[:3]:
                arrow = "+" if d["shap_value"] > 0 else "-"
                print(f"             {arrow} {d['feature']:30s} {d['shap_value']:+.4f}")

    print("\n" + "=" * 60)
    print("  Dashboard: streamlit run app/dashboard.py")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    t_start = time.time()

    log.info("=" * 55)
    log.info("  IBEX35 Morning Run — " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 55)

    # ── 1. Retrain (optional, weekly) ────────────────────────────────────────
    if args.retrain:
        log.info("Step 1/3: Retraining all models (this takes ~15-30 min)...")
        rc = _run([sys.executable, "train.py", "--all-targets", "--force-download"],
                  "Model retraining")
        if rc != 0:
            log.error("Retraining failed — aborting morning run.")
            sys.exit(rc)
    else:
        log.info("Step 1/3: Skipping retraining (use --retrain to force)")

    # ── 2. Generate prediction ────────────────────────────────────────────────
    log.info("Step 2/3: Generating prediction (force-downloading latest data)...")
    predict_cmd = [sys.executable, "predict.py", "--force-download"]
    if not args.no_shap:
        predict_cmd.append("--shap")

    rc = _run(predict_cmd, "Prediction")
    if rc != 0:
        log.error("Prediction failed.")
        sys.exit(rc)

    # ── 3. Generate operative summary ─────────────────────────────────────────
    if not args.no_summary:
        log.info("Step 3/3: Generating operative summary...")
        try:
            from app.summary import generate_summary, save_summary
            summary = generate_summary(include_live_news=True)
            out = save_summary(summary)
            log.info(f"Summary saved -> {out}")
        except Exception as exc:
            log.warning(f"Summary generation failed (non-fatal): {exc}")
    else:
        log.info("Step 3/3: Skipping summary (--no-summary)")

    # ── Print morning briefing ────────────────────────────────────────────────
    _print_signal_banner(root() / "results" / "latest_prediction.json")

    elapsed = time.time() - t_start
    log.info(f"Morning run complete in {elapsed:.0f}s")

    # ── 4. Open dashboard (optional) ─────────────────────────────────────────
    if args.open_dashboard:
        log.info("Launching dashboard at http://localhost:8501 ...")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"],
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
