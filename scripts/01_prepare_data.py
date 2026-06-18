from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import main as build_dataset_main  # noqa: E402
from src.data.knowledge_prior import build_knowledge_prior  # noqa: E402
from src.data.preprocessing import create_all_fold_preprocess  # noqa: E402
from src.data.prior_alignment import write_prior_alignment_outputs  # noqa: E402
from src.data.split import create_splits  # noqa: E402


def main() -> int:
    summary = build_dataset_main(["--project-root", str(PROJECT_ROOT)])
    if summary.get("built_sample_count") != 417:
        raise RuntimeError(f"Expected 417 samples, got {summary.get('built_sample_count')}")
    split_summary = create_splits(PROJECT_ROOT)
    logging.info("Split summary: %s", split_summary)
    preprocess_paths = create_all_fold_preprocess(PROJECT_ROOT)
    logging.info("Created fold preprocessing objects: %s", [str(path) for path in preprocess_paths])
    prior = build_knowledge_prior(PROJECT_ROOT)
    logging.info(
        "Knowledge prior built: treatment_biomarker=%s biomarker_biomarker=%s",
        prior["treatment_biomarker_prior"].shape,
        prior["biomarker_biomarker_prior"].shape,
    )
    prior_alignment = write_prior_alignment_outputs(PROJECT_ROOT)
    logging.info("Prior alignment audit: %s", prior_alignment)
    if not prior_alignment.get("all_checks_passed"):
        raise RuntimeError("Prior alignment audit failed; refusing to continue.")
    logging.info("Data preparation completed. No full training or baseline run was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
