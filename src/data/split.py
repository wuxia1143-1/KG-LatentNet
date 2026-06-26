from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 20260605
PATIENT_ID_COL = "patient_SN"
STRATIFY_COL = "endpoint_window"


def distribution(rows: pd.DataFrame) -> dict[str, int]:
    counts = Counter(int(value) for value in rows[STRATIFY_COL].tolist())
    return {str(window): int(counts.get(window, 0)) for window in [6, 12, 18, 24]}


def overlap_count(left: list[str], right: list[str]) -> int:
    return len(set(left) & set(right))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_splits(project_root: Path, seed: int = SEED) -> dict[str, int]:
    processed_path = project_root / "data" / "processed" / "patient_level_samples.csv"
    splits_dir = project_root / "data" / "splits"
    tables_dir = project_root / "results" / "tables"
    splits_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(processed_path)
    if samples[PATIENT_ID_COL].nunique() != len(samples):
        raise ValueError("Patient-level samples contain duplicate patient_SN values.")
    if len(samples) != 417:
        raise ValueError(f"Expected 417 patient-level samples before splitting, got {len(samples)}.")

    distribution_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[pd.DataFrame] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (dev_idx, test_idx) in enumerate(skf.split(samples[PATIENT_ID_COL], samples[STRATIFY_COL])):
        dev = samples.iloc[dev_idx].copy()
        test = samples.iloc[test_idx].copy()
        train, val = train_test_split(
            dev,
            test_size=0.10,
            random_state=seed + fold,
            stratify=dev[STRATIFY_COL],
        )
        train_ids = train[PATIENT_ID_COL].astype(str).tolist()
        val_ids = val[PATIENT_ID_COL].astype(str).tolist()
        test_ids = test[PATIENT_ID_COL].astype(str).tolist()
        payload = {
            "fold": fold,
            "seed": seed,
            "unit": PATIENT_ID_COL,
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
            "test_patient_ids": test_ids,
            "distribution": {
                "train": distribution(train),
                "val": distribution(val),
                "test": distribution(test),
            },
            "counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
        }
        write_json(splits_dir / f"fold_{fold}.json", payload)
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            counts = distribution(split_df)
            row = {"fold": fold, "split": split_name, "n": len(split_df)}
            row.update({f"window_{window}": counts[str(window)] for window in [6, 12, 18, 24]})
            distribution_rows.append(row)
        train_val_overlap = overlap_count(train_ids, val_ids)
        train_test_overlap = overlap_count(train_ids, test_ids)
        val_test_overlap = overlap_count(val_ids, test_ids)
        leakage_rows.append(
            {
                "fold": fold,
                "train_val_overlap": train_val_overlap,
                "train_test_overlap": train_test_overlap,
                "val_test_overlap": val_test_overlap,
                "patient_level_leakage": int(any([train_val_overlap, train_test_overlap, val_test_overlap])),
                "endpoint_tbr_in_features": 0,
            }
        )
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            tmp = split_df[[PATIENT_ID_COL, STRATIFY_COL]].copy()
            tmp["fold"] = fold
            tmp["split"] = split_name
            fold_assignment_rows.append(tmp)

    pd.DataFrame(distribution_rows).to_csv(tables_dir / "fold_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leakage_rows).to_csv(tables_dir / "fold_leakage_check.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_assignment_rows, ignore_index=True).to_csv(splits_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")

    summary = {
        "folds": 5,
        "patients": len(samples),
        "patient_level_leakage": int(any(row["patient_level_leakage"] for row in leakage_rows)),
    }
    logging.info("Created patient-level 5-fold splits: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description="Create KG_LatentNet patient-level splits.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    return create_splits(Path(args.project_root).resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
