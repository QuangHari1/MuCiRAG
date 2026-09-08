"""Start the local MLflow UI against this baseline's persistent experiment store."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DATABASE = PROJECT_ROOT / "MuCiRAG/results/mlflow/mlflow.db"
PUBLIC_HOSTNAME = "mlflow.quanghari.uk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not TRACKING_DATABASE.is_file():
        raise FileNotFoundError(
            f"MLflow store is absent: {TRACKING_DATABASE}. "
            "Run the TeleQnA benchmark with experiment_tracking.mode = \"mlflow\" first."
        )
    backend_uri = f"sqlite:///{TRACKING_DATABASE}"
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--allowed-hosts",
        PUBLIC_HOSTNAME,
        "--cors-allowed-origins",
        f"https://{PUBLIC_HOSTNAME}",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
