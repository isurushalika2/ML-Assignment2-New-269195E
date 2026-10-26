"""
Main entry point for the Ratmalana Next-Day Rainfall Prediction & XAI Project.
- Runs the benchmark, training, and XAI artifact generation pipeline if artifacts are missing.
- Launches the Streamlit frontend web application when executed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from model_selection_benchmark import main as run_benchmark_pipeline

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "artifacts" / "models_bundle.joblib"


def main():
    if not BUNDLE_PATH.exists():
        print("Training models and generating XAI artifacts...")
        run_benchmark_pipeline()
    else:
        print(f"Found pre-trained model bundle at: {BUNDLE_PATH}")

    print("\nLaunching Streamlit Frontend Dashboard (http://localhost:8501)...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")],
        check=False,
    )


if __name__ == "__main__":
    main()
