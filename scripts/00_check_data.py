from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import main as build_dataset_main  # noqa: E402
from src.data.split import create_splits  # noqa: E402


def main() -> int:
    summary = build_dataset_main(["--project-root", str(PROJECT_ROOT)])
    if summary.get("built_sample_count") == 417:
        split_summary = create_splits(PROJECT_ROOT)
        logging.info("Split summary: %s", split_summary)
    else:
        logging.warning("Skipping split creation because built_sample_count != 417.")
    logging.info("Data check completed. No model training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
