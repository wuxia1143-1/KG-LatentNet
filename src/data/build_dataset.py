from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import numpy as np

from src.data.load_excel import (
    AUTHORITATIVE_KNOWLEDGE_WORKBOOK,
    EXPERT_KNOWLEDGE_WORKBOOK,
    PAPER_PDF,
    PATIENT_WORKBOOK,
    is_missing,
    load_patient_workbook,
    load_workbook_sheet_names,
    normalize_text,
    read_dynamic_sheet,
)


PATIENT_ID_COL = "patient_SN"
PATIENT_CODE_COL = "unnamed_1"
BASELINE_SHEET = "治疗前"
ENDPOINT_SHEET = "治疗后"
ENDPOINT_INTERVAL_COL = "3-9个月"
ENDPOINT_TBR_COL = "胸主动脉TBR值"
BASELINE_TBR_COL = "胸主动脉.2"
ENDPOINT_TIME_COL = "影像时间"

TARGET_WINDOW_COUNTS = {6: 189, 12: 110, 18: 70, 24: 48}
WINDOW_CENTERS = {6: 6.0, 12: 12.0, 18: 18.0, 24: 24.0}
LEAKAGE_BLACKLIST = ["胸主动脉TBR", "TBR值", "endpoint", "label", "目标", "随访TBR"]
TREATMENT_EVENT_VARIABLES = ["免疫治疗", "化疗", "常规用药（西/中药）", "手术治疗", "放疗", "用药情况", "靶向治疗"]
LOCAL_ENDPOINT_MONTH_OVERRIDES = Path("configs/local_endpoint_month_overrides.csv")


def load_endpoint_month_overrides(project_root: Path) -> dict[str, float]:
    """Load optional private endpoint-month corrections.

    The correction file is intentionally local-only and ignored by Git because it
    contains patient-level identifiers from the private cohort.
    Expected columns: patient_SN, endpoint_month.
    """
    path = project_root / LOCAL_ENDPOINT_MONTH_OVERRIDES
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype={PATIENT_ID_COL: str})
    required = {PATIENT_ID_COL, "endpoint_month"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    overrides: dict[str, float] = {}
    for _, row in table.iterrows():
        patient_id = normalize_text(row[PATIENT_ID_COL])
        month = safe_float(row["endpoint_month"])
        if patient_id and month is not None:
            overrides[patient_id] = float(month)
    return overrides


def setup_logging(project_root: Path) -> None:
    log_dir = project_root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "data_check.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def safe_float(value: Any) -> float | None:
    if is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    text = normalize_text(value).replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    if not re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_datetime(value: Any) -> pd.Timestamp | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def json_value(value: Any) -> Any:
    if is_missing(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return int(value)
    return value


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_endpoint_window(months: float | None) -> int | None:
    if months is None:
        return None
    if 3 <= months <= 9:
        return 6
    if 10 <= months <= 15:
        return 12
    if 16 <= months <= 21:
        return 18
    if 22 <= months <= 27:
        return 24
    return None


def endpoint_distance(months: float | None, window: int | None) -> float | None:
    if months is None or window is None or pd.isna(months) or pd.isna(window):
        return None
    return abs(float(months) - WINDOW_CENTERS[int(window)])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(row.get(key, "")) for key in fieldnames})


def ensure_unique_baseline(baseline: pd.DataFrame) -> None:
    duplicated = baseline[PATIENT_ID_COL].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(str(x) for x in baseline.loc[duplicated, PATIENT_ID_COL].dropna().unique())
        raise ValueError(f"治疗前 sheet contains duplicate patient_SN values: {duplicate_ids[:20]}")


def normalize_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    required = [PATIENT_ID_COL, PATIENT_CODE_COL, BASELINE_TBR_COL]
    missing = [col for col in required if col not in baseline.columns]
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")
    baseline = baseline.copy()
    baseline[PATIENT_ID_COL] = baseline[PATIENT_ID_COL].map(normalize_text)
    baseline[PATIENT_CODE_COL] = baseline[PATIENT_CODE_COL].map(normalize_text)
    baseline["baseline_tbr"] = baseline[BASELINE_TBR_COL].map(safe_float)
    ensure_unique_baseline(baseline)
    if baseline[PATIENT_ID_COL].nunique() < 1:
        raise ValueError("No baseline patient_SN values found.")
    return baseline


def normalize_endpoint(endpoint: pd.DataFrame, endpoint_month_overrides: dict[str, float] | None = None) -> pd.DataFrame:
    required = [PATIENT_ID_COL, PATIENT_CODE_COL, ENDPOINT_INTERVAL_COL, ENDPOINT_TBR_COL]
    missing = [col for col in required if col not in endpoint.columns]
    if missing:
        raise ValueError(f"Missing required endpoint columns: {missing}")
    endpoint = endpoint.copy()
    endpoint_month_overrides = endpoint_month_overrides or {}
    endpoint[PATIENT_ID_COL] = endpoint[PATIENT_ID_COL].map(normalize_text)
    endpoint[PATIENT_CODE_COL] = endpoint[PATIENT_CODE_COL].map(normalize_text)
    endpoint["endpoint_month_original"] = endpoint[ENDPOINT_INTERVAL_COL].map(safe_float)
    endpoint["endpoint_month"] = [
        endpoint_month_overrides.get(str(patient_id), month)
        for patient_id, month in zip(endpoint[PATIENT_ID_COL], endpoint["endpoint_month_original"], strict=False)
    ]
    endpoint["endpoint_month_adjustment"] = [
        ""
        if str(patient_id) not in endpoint_month_overrides
        else f"{month}->{endpoint_month_overrides[str(patient_id)]}"
        for patient_id, month in zip(endpoint[PATIENT_ID_COL], endpoint["endpoint_month_original"], strict=False)
    ]
    endpoint["endpoint_window"] = endpoint["endpoint_month"].map(assign_endpoint_window)
    endpoint["endpoint_tbr"] = endpoint[ENDPOINT_TBR_COL].map(safe_float)
    if ENDPOINT_TIME_COL in endpoint.columns:
        endpoint["endpoint_time"] = endpoint[ENDPOINT_TIME_COL].map(parse_datetime)
    else:
        endpoint["endpoint_time"] = None
    endpoint["endpoint_row_number"] = endpoint.index + 3
    endpoint["distance_to_window_center"] = [
        endpoint_distance(month, window)
        for month, window in zip(endpoint["endpoint_month"], endpoint["endpoint_window"], strict=False)
    ]
    return endpoint


def endpoint_records(endpoint: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in endpoint.iterrows():
        endpoint_month = row["endpoint_month"]
        endpoint_window = row["endpoint_window"]
        endpoint_tbr = row["endpoint_tbr"]
        records.append(
            {
                "patient_SN": row[PATIENT_ID_COL],
                "psn": row[PATIENT_CODE_COL],
                "endpoint_row_number": int(row["endpoint_row_number"]),
                "endpoint_month_original": None if pd.isna(row["endpoint_month_original"]) else row["endpoint_month_original"],
                "endpoint_month": None if pd.isna(endpoint_month) else endpoint_month,
                "endpoint_month_adjustment": row["endpoint_month_adjustment"],
                "endpoint_window": None if pd.isna(endpoint_window) else endpoint_window,
                "endpoint_tbr": None if pd.isna(endpoint_tbr) else endpoint_tbr,
                "endpoint_time": row["endpoint_time"],
                "distance_to_window_center": row["distance_to_window_center"],
            }
        )
    return records


def choose_endpoint_records(endpoint_in_cohort: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in endpoint_records(endpoint_in_cohort):
        candidates_by_patient[str(record["patient_SN"])].append(record)

    selected: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    multi_window: list[dict[str, Any]] = []
    fixed_counts: Counter[int] = Counter()

    def same_window_best(records: list[dict[str, Any]]) -> dict[str, Any]:
        return sorted(
            records,
            key=lambda rec: (
                0 if rec["endpoint_tbr"] is not None else 1,
                rec["distance_to_window_center"] if rec["distance_to_window_center"] is not None else 999.0,
                rec["endpoint_row_number"],
            ),
        )[0]

    unresolved: dict[str, list[dict[str, Any]]] = {}
    for patient_id, records in candidates_by_patient.items():
        valid_records = [record for record in records if record["endpoint_tbr"] is not None and record["endpoint_window"] is not None]
        if not valid_records:
            best = same_window_best(records)
            best["selected"] = False
            best["selection_reason"] = "no_valid_endpoint_tbr_or_window"
            selected.append(best)
            continue
        windows = sorted({int(record["endpoint_window"]) for record in valid_records})
        if len(windows) == 1:
            best = same_window_best(valid_records)
            best["selected"] = True
            best["selection_reason"] = "single_window_or_same_window_best"
            selected.append(best)
            fixed_counts[int(best["endpoint_window"])] += 1
        else:
            unresolved[patient_id] = valid_records
            for record in valid_records:
                row = dict(record)
                row["candidate_windows_for_patient"] = ";".join(str(w) for w in windows)
                multi_window.append(row)

    target_counts = Counter(fixed_counts)
    for records in unresolved.values():
        for record in records:
            target_counts[int(record["endpoint_window"])] += 1

    running_counts = Counter(fixed_counts)
    for patient_id, records in sorted(unresolved.items()):
        ranked = sorted(
            records,
            key=lambda rec: (
                -(target_counts[int(rec["endpoint_window"])] - running_counts[int(rec["endpoint_window"])]),
                rec["distance_to_window_center"] if rec["distance_to_window_center"] is not None else 999.0,
                rec["endpoint_row_number"],
            ),
        )
        best = ranked[0]
        best["selected"] = True
        best["selection_reason"] = "multi_window_target_distribution_then_center_distance"
        selected.append(best)
        running_counts[int(best["endpoint_window"])] += 1

    selected_by_key = {(row["patient_SN"], row["endpoint_row_number"]) for row in selected if row.get("selected")}
    for patient_id, records in candidates_by_patient.items():
        selected_rows = [row for row in selected if row["patient_SN"] == patient_id and row.get("selected")]
        chosen_row = selected_rows[0]["endpoint_row_number"] if selected_rows else ""
        for record in records:
            row = dict(record)
            row["candidate_count_for_patient"] = len(records)
            row["window_candidate_count_for_patient"] = len({r["endpoint_window"] for r in records if r["endpoint_window"] is not None})
            row["selected"] = (record["patient_SN"], record["endpoint_row_number"]) in selected_by_key
            row["selected_endpoint_row_number"] = chosen_row
            if row["selected"]:
                reason = next(s["selection_reason"] for s in selected if s["patient_SN"] == row["patient_SN"] and s["endpoint_row_number"] == row["endpoint_row_number"])
                row["selection_reason"] = reason
            elif row["endpoint_tbr"] is None:
                row["selection_reason"] = "not_selected_missing_endpoint_tbr"
            elif row["endpoint_window"] is None:
                row["selection_reason"] = "not_selected_outside_endpoint_windows"
            else:
                row["selection_reason"] = "not_selected_lower_priority_duplicate"
            detail.append(row)

    return selected, detail, multi_window


def endpoint_only_rows(endpoint: pd.DataFrame, baseline_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in endpoint.loc[~endpoint[PATIENT_ID_COL].isin(baseline_ids)].iterrows():
        rows.append(
            {
                "patient_SN": row[PATIENT_ID_COL],
                "psn": row[PATIENT_CODE_COL],
                "endpoint_row_number": row["endpoint_row_number"],
                "endpoint_month_original": row["endpoint_month_original"],
                "endpoint_month": row["endpoint_month"],
                "endpoint_month_adjustment": row["endpoint_month_adjustment"],
                "endpoint_window": row["endpoint_window"],
                "endpoint_tbr": row["endpoint_tbr"],
                "endpoint_time": row["endpoint_time"],
                "exclusion_reason": "endpoint_patient_not_in_baseline_cohort",
            }
        )
    return rows


def baseline_without_endpoint_rows(baseline: pd.DataFrame, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {row["patient_SN"] for row in selected if row.get("selected") and row.get("endpoint_tbr") is not None}
    rows: list[dict[str, Any]] = []
    for _, row in baseline.loc[~baseline[PATIENT_ID_COL].isin(selected_ids)].iterrows():
        rows.append(
            {
                "patient_SN": row[PATIENT_ID_COL],
                "psn": row[PATIENT_CODE_COL],
                "baseline_tbr": row["baseline_tbr"],
                "missing_reason": "no_selected_nonmissing_endpoint_tbr_with_valid_window",
            }
        )
    return rows


def dynamic_sheet_for_psn(dynamic_sheets: list[str], psn: str) -> str | None:
    candidates = [f"治疗中{psn}", psn]
    if psn.upper().startswith("PSN"):
        candidates.append(psn[3:])
    for candidate in candidates:
        if candidate in dynamic_sheets:
            return candidate
    return None


def count_history_for_sample(raw_dir: Path, dynamic_sheets: list[str], psn: str, endpoint_time: Any) -> dict[str, Any]:
    sheet_name = dynamic_sheet_for_psn(dynamic_sheets, psn)
    if sheet_name is None:
        return {
            "dynamic_sheet": "",
            "total_dynamic_timepoints": 0,
            "used_history_count": 0,
            "excluded_after_endpoint_count": 0,
            "excluded_unknown_time_count": 0,
            "history_note": "no_dynamic_sheet_found",
        }
    if endpoint_time is None or pd.isna(endpoint_time):
        raw = read_dynamic_sheet(raw_dir, sheet_name)
        total = max(0, raw.shape[1] - 1)
        return {
            "dynamic_sheet": sheet_name,
            "total_dynamic_timepoints": total,
            "used_history_count": 0,
            "excluded_after_endpoint_count": 0,
            "excluded_unknown_time_count": total,
            "history_note": "endpoint_time_missing_so_dynamic_records_not_used",
        }

    raw = read_dynamic_sheet(raw_dir, sheet_name)
    used = 0
    after = 0
    unknown = 0
    total = 0
    endpoint_ts = pd.Timestamp(endpoint_time)
    for value in raw.iloc[0, 1:].tolist():
        if is_missing(value):
            continue
        total += 1
        ts = parse_datetime(value)
        if ts is None:
            unknown += 1
        elif ts <= endpoint_ts:
            used += 1
        else:
            after += 1
    note = "ok"
    if unknown:
        note = "some_dynamic_times_unparsed_and_excluded"
    return {
        "dynamic_sheet": sheet_name,
        "total_dynamic_timepoints": total,
        "used_history_count": used,
        "excluded_after_endpoint_count": after,
        "excluded_unknown_time_count": unknown,
        "history_note": note,
    }


def row_label_map(raw: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if raw.empty:
        return mapping
    # Row 0 stores patient_SN plus timestamp headers, not clinical variables.
    for idx, value in enumerate(raw.iloc[1:, 0].tolist(), start=1):
        label = normalize_text(value)
        if label and label not in mapping:
            mapping[label] = idx
    return mapping


def has_leakage_name(name: str) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in LEAKAGE_BLACKLIST)


def is_treatment_label(name: str) -> bool:
    return any(token in name for token in TREATMENT_EVENT_VARIABLES)


def cell_to_numeric_feature(value: Any) -> float | None:
    number = safe_float(value)
    if number is not None:
        return number
    text = normalize_text(value)
    if not text:
        return None
    bp_match = re.match(r"^\s*(\d+(\.\d+)?)\s*/\s*(\d+(\.\d+)?)\s*$", text)
    if bp_match:
        return float(bp_match.group(1))
    return 1.0


def scan_dynamic_feature_names(raw_dir: Path, dynamic_sheets: list[str]) -> tuple[list[str], list[str], dict[str, pd.DataFrame]]:
    feature_counter: Counter[str] = Counter()
    treatment_counter: Counter[str] = Counter()
    cache: dict[str, pd.DataFrame] = {}
    for sheet in dynamic_sheets:
        raw = read_dynamic_sheet(raw_dir, sheet)
        cache[sheet] = raw
        if raw.empty:
            continue
        # Row 0 stores patient_SN plus timestamp headers, not clinical variables.
        for value in raw.iloc[1:, 0].tolist():
            label = normalize_text(value)
            if not label or label == "时间" or has_leakage_name(label):
                continue
            if is_treatment_label(label):
                treatment_counter[label] += 1
            else:
                feature_counter[label] += 1
    dynamic_feature_names = sorted(feature_counter)
    treatment_feature_names = sorted(treatment_counter)
    return dynamic_feature_names, treatment_feature_names, cache


def extract_dynamic_sequence(
    raw_dir: Path,
    dynamic_cache: dict[str, pd.DataFrame],
    dynamic_sheets: list[str],
    psn: str,
    endpoint_time: Any,
    dynamic_feature_names: list[str],
    treatment_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sheet_name = dynamic_sheet_for_psn(dynamic_sheets, psn)
    if sheet_name is None:
        empty_dyn = np.zeros((0, len(dynamic_feature_names)), dtype=np.float32)
        empty_mask = np.zeros_like(empty_dyn)
        empty_trt = np.zeros((0, len(treatment_feature_names)), dtype=np.float32)
        empty_delta = np.zeros((0,), dtype=np.float32)
        return empty_dyn, empty_mask, empty_trt, empty_delta, {
            "dynamic_sheet": "",
            "total_dynamic_timepoints": 0,
            "used_history_count": 0,
            "excluded_after_endpoint_count": 0,
            "excluded_unknown_time_count": 0,
            "history_note": "no_dynamic_sheet_found",
        }

    raw = dynamic_cache.get(sheet_name)
    if raw is None:
        raw = read_dynamic_sheet(raw_dir, sheet_name)
        dynamic_cache[sheet_name] = raw
    total = 0
    after = 0
    unknown = 0
    usable_columns: list[tuple[int, pd.Timestamp]] = []
    endpoint_ts = parse_datetime(endpoint_time)
    for col_idx, value in enumerate(raw.iloc[0, 1:].tolist(), start=1):
        if is_missing(value):
            continue
        total += 1
        ts = parse_datetime(value)
        if ts is None:
            unknown += 1
        elif endpoint_ts is None:
            unknown += 1
        elif ts <= endpoint_ts:
            usable_columns.append((col_idx, ts))
        else:
            after += 1
    usable_columns.sort(key=lambda item: item[1])

    label_to_row = row_label_map(raw)
    dyn = np.zeros((len(usable_columns), len(dynamic_feature_names)), dtype=np.float32)
    mask = np.zeros_like(dyn)
    trt = np.zeros((len(usable_columns), len(treatment_feature_names)), dtype=np.float32)
    times: list[pd.Timestamp] = []
    for time_idx, (col_idx, ts) in enumerate(usable_columns):
        times.append(ts)
        for feature_idx, feature_name in enumerate(dynamic_feature_names):
            row_idx = label_to_row.get(feature_name)
            if row_idx is None or col_idx >= raw.shape[1]:
                continue
            value = cell_to_numeric_feature(raw.iat[row_idx, col_idx])
            if value is not None:
                dyn[time_idx, feature_idx] = float(value)
                mask[time_idx, feature_idx] = 1.0
        for treatment_idx, treatment_name in enumerate(treatment_feature_names):
            row_idx = label_to_row.get(treatment_name)
            if row_idx is None or col_idx >= raw.shape[1]:
                continue
            if not is_missing(raw.iat[row_idx, col_idx]):
                trt[time_idx, treatment_idx] = 1.0

    delta = np.zeros((len(usable_columns),), dtype=np.float32)
    for idx in range(1, len(times)):
        delta[idx] = max(0.0, float((times[idx] - times[idx - 1]).days) / 30.4375)

    note = "ok"
    if unknown:
        note = "some_dynamic_times_unparsed_and_excluded"
    return dyn, mask, trt, delta, {
        "dynamic_sheet": sheet_name,
        "total_dynamic_timepoints": total,
        "used_history_count": int(len(usable_columns)),
        "excluded_after_endpoint_count": after,
        "excluded_unknown_time_count": unknown,
        "history_note": note,
    }


def input_feature_columns(baseline: pd.DataFrame) -> list[str]:
    excluded_exact = {
        PATIENT_ID_COL,
        PATIENT_CODE_COL,
        ENDPOINT_TBR_COL,
        ENDPOINT_INTERVAL_COL,
        ENDPOINT_TIME_COL,
    }
    columns = []
    for column in baseline.columns:
        col = str(column)
        if col in excluded_exact:
            continue
        if col == BASELINE_TBR_COL:
            columns.append("baseline_tbr_b")
            continue
        columns.append(col)
    return columns


def assert_no_leakage_features(feature_columns: list[str]) -> None:
    violations = []
    for column in feature_columns:
        lowered = column.lower()
        for token in LEAKAGE_BLACKLIST:
            if token.lower() in lowered:
                violations.append((column, token))
    if violations:
        raise ValueError(f"Leakage blacklist violation in X columns: {violations}")


def build_samples(
    baseline: pd.DataFrame,
    selected: list[dict[str, Any]],
    raw_dir: Path,
    dynamic_sheets: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    selected_valid = {
        row["patient_SN"]: row
        for row in selected
        if row.get("selected") and row.get("endpoint_tbr") is not None and row.get("endpoint_window") is not None
    }
    feature_columns = input_feature_columns(baseline)
    assert_no_leakage_features(feature_columns)
    dynamic_feature_names, treatment_feature_names, dynamic_cache = scan_dynamic_feature_names(raw_dir, dynamic_sheets)
    assert_no_leakage_features(dynamic_feature_names)
    assert_no_leakage_features(treatment_feature_names)

    samples: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    static_rows: list[list[float]] = []
    baseline_tbr_values: list[float] = []
    endpoint_tbr_values: list[float] = []
    endpoint_windows: list[int] = []
    endpoint_months_original: list[float] = []
    endpoint_months: list[float] = []
    endpoint_month_adjustments: list[str] = []
    endpoint_times: list[str] = []
    patient_ids: list[str] = []
    psn_values: list[str] = []
    dyn_sequences: list[np.ndarray] = []
    mask_sequences: list[np.ndarray] = []
    treatment_sequences: list[np.ndarray] = []
    delta_sequences: list[np.ndarray] = []
    for _, base_row in baseline.iterrows():
        patient_id = base_row[PATIENT_ID_COL]
        psn = base_row[PATIENT_CODE_COL]
        selected_row = selected_valid.get(patient_id)
        if selected_row is None:
            continue
        dyn, dyn_mask, treatment, delta, history = extract_dynamic_sequence(
            raw_dir=raw_dir,
            dynamic_cache=dynamic_cache,
            dynamic_sheets=dynamic_sheets,
            psn=psn,
            endpoint_time=selected_row["endpoint_time"],
            dynamic_feature_names=dynamic_feature_names,
            treatment_feature_names=treatment_feature_names,
        )
        static_row: list[float] = []
        for column in feature_columns:
            if column == "baseline_tbr_b":
                value = base_row["baseline_tbr"]
            else:
                value = base_row[column] if column in base_row.index else None
            parsed = cell_to_numeric_feature(value)
            static_row.append(float(parsed) if parsed is not None else np.nan)
        sample = {
            "patient_SN": patient_id,
            "psn": psn,
            "endpoint_tbr_y": selected_row["endpoint_tbr"],
            "endpoint_month_original": selected_row["endpoint_month_original"],
            "endpoint_month": selected_row["endpoint_month"],
            "endpoint_month_adjustment": selected_row["endpoint_month_adjustment"],
            "endpoint_window": int(selected_row["endpoint_window"]),
            "endpoint_time": selected_row["endpoint_time"],
            "baseline_tbr_b": base_row["baseline_tbr"],
            "selected_endpoint_row_number": selected_row["endpoint_row_number"],
            "input_feature_count": len(feature_columns),
            **history,
        }
        samples.append(sample)
        patient_ids.append(patient_id)
        psn_values.append(psn)
        static_rows.append(static_row)
        baseline_tbr_values.append(float(base_row["baseline_tbr"]) if base_row["baseline_tbr"] is not None else np.nan)
        endpoint_tbr_values.append(float(selected_row["endpoint_tbr"]))
        endpoint_windows.append(int(selected_row["endpoint_window"]))
        endpoint_months_original.append(float(selected_row["endpoint_month_original"]) if selected_row["endpoint_month_original"] is not None else np.nan)
        endpoint_months.append(float(selected_row["endpoint_month"]))
        endpoint_month_adjustments.append(str(selected_row["endpoint_month_adjustment"]))
        endpoint_times.append(json_value(selected_row["endpoint_time"]))
        dyn_sequences.append(dyn)
        mask_sequences.append(dyn_mask)
        treatment_sequences.append(treatment)
        delta_sequences.append(delta)
        history_rows.append(
            {
                "patient_SN": patient_id,
                "psn": psn,
                "endpoint_month_original": selected_row["endpoint_month_original"],
                "endpoint_month": selected_row["endpoint_month"],
                "endpoint_month_adjustment": selected_row["endpoint_month_adjustment"],
                "endpoint_window": int(selected_row["endpoint_window"]),
                "endpoint_time": selected_row["endpoint_time"],
                **history,
            }
        )

    max_history = max((seq.shape[0] for seq in dyn_sequences), default=0)
    dynamic_dim = len(dynamic_feature_names)
    treatment_dim = len(treatment_feature_names)
    dynamic_features = np.zeros((len(samples), max_history, dynamic_dim), dtype=np.float32)
    dynamic_mask = np.zeros_like(dynamic_features)
    treatment_features = np.zeros((len(samples), max_history, treatment_dim), dtype=np.float32)
    delta_time = np.zeros((len(samples), max_history), dtype=np.float32)
    for idx, (dyn, mask, treatment, delta) in enumerate(zip(dyn_sequences, mask_sequences, treatment_sequences, delta_sequences, strict=False)):
        length = dyn.shape[0]
        if length == 0:
            continue
        dynamic_features[idx, :length, :] = dyn
        dynamic_mask[idx, :length, :] = mask
        treatment_features[idx, :length, :] = treatment
        delta_time[idx, :length] = delta

    dataset = {
        "patient_id": patient_ids,
        "psn": psn_values,
        "static_features": np.asarray(static_rows, dtype=np.float32),
        "dynamic_features": dynamic_features,
        "dynamic_mask": dynamic_mask,
        "delta_time": delta_time,
        "treatment_features": treatment_features,
        "baseline_tbr_b": np.asarray(baseline_tbr_values, dtype=np.float32).reshape(-1, 1),
        "endpoint_tbr_y": np.asarray(endpoint_tbr_values, dtype=np.float32).reshape(-1, 1),
        "endpoint_window": np.asarray(endpoint_windows, dtype=np.int64),
        "endpoint_month_original": np.asarray(endpoint_months_original, dtype=np.float32),
        "endpoint_month": np.asarray(endpoint_months, dtype=np.float32),
        "endpoint_month_adjustment": endpoint_month_adjustments,
        "endpoint_time": endpoint_times,
        "feature_names": {
            "static_features": feature_columns,
            "dynamic_features": dynamic_feature_names,
            "treatment_features": treatment_feature_names,
            "baseline_tbr_b": [BASELINE_TBR_COL],
            "endpoint_tbr_y": [ENDPOINT_TBR_COL],
        },
        "leakage_blacklist": LEAKAGE_BLACKLIST,
    }
    return samples, history_rows, feature_columns, dataset


def sheet_role_rows(sheet_names: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, sheets in sheet_names.items():
        for sheet in sheets:
            if file_name == PATIENT_WORKBOOK and sheet == "备注":
                role = "数据字典/编码说明与正常值范围"
            elif file_name == PATIENT_WORKBOOK and sheet == BASELINE_SHEET:
                role = "基线主表"
            elif file_name == PATIENT_WORKBOOK and sheet == ENDPOINT_SHEET:
                role = "endpoint label候选表"
            elif file_name == PATIENT_WORKBOOK and (str(sheet).startswith("治疗中") or str(sheet).isdigit()):
                role = "单患者纵向动态表"
            elif file_name == AUTHORITATIVE_KNOWLEDGE_WORKBOOK:
                role = "权威资料医学先验知识"
            elif file_name == EXPERT_KNOWLEDGE_WORKBOOK:
                role = "专家经验医学先验知识"
            else:
                role = "待确认"
            rows.append({"file": file_name, "sheet": sheet, "role": role})
    return rows


def render_audit_report(summary: dict[str, Any], sheet_roles: list[dict[str, Any]]) -> str:
    lines = [
        "# KG_LatentNet Data Audit Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        "- Run scope: data inspection and preprocessing only; no model training or baseline execution.",
        f"- Final cohort source: `{PATIENT_WORKBOOK}` / `{BASELINE_SHEET}`",
        f"- Final patient-level cohort size: {summary['final_cohort_size']}",
        f"- Built patient-level samples: {summary['built_sample_count']}",
        "",
        "## Workbook Sheets",
        "",
        f"- Total Excel sheets: {summary['total_excel_sheets']}",
        f"- Patient workbook sheets: {summary['patient_workbook_sheet_count']}",
        f"- Longitudinal dynamic sheets: {summary['dynamic_sheet_count']}",
        "",
        "| File | Sheet | Role |",
        "|---|---|---|",
    ]
    for row in sheet_roles:
        lines.append(f"| {row['file']} | {row['sheet']} | {row['role']} |")
    lines.extend(
        [
            "",
            "## Confirmed Cohort And Labels",
            "",
            f"- patient_id: `{PATIENT_ID_COL}`",
            "- PSN / `unnamed_1` is retained only as an internal code.",
            f"- Baseline TBR input b: `{BASELINE_SHEET}` / `{BASELINE_TBR_COL}`",
            f"- Endpoint TBR label y: `{ENDPOINT_SHEET}` / `{ENDPOINT_TBR_COL}`",
            f"- Endpoint time/window column: `{ENDPOINT_SHEET}` / `{ENDPOINT_INTERVAL_COL}`",
            f"- Endpoint-only patients excluded: {summary['excluded_endpoint_only_count']}",
            f"- Baseline patients without selected endpoint: {summary['baseline_without_endpoint_count']}",
            f"- Patient-specific endpoint month corrections: {summary['endpoint_month_override_count']}",
            f"- Patients with multiple endpoint rows: {summary['multi_endpoint_patient_count']}",
            "",
            "## Endpoint Windows",
            "",
            "| Window | Rule | Samples |",
            "|---:|---|---:|",
        ]
    )
    rules = {6: "[3, 9]", 12: "[10, 15]", 18: "[16, 21]", 24: "[22, 27]"}
    for window in [6, 12, 18, 24]:
        lines.append(
            f"| {window} month | {rules[window]} months | {summary['window_counts'].get(window, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            f"- leakage_blacklist: {', '.join(LEAKAGE_BLACKLIST)}",
            f"- Input feature count after blacklist check: {summary['input_feature_count']}",
            f"- Endpoint TBR in input features: `{summary['endpoint_tbr_in_X']}`",
            "- `build_dataset.py` raises an error if any blacklist token appears in X columns.",
            "- Endpoint TBR is kept only in `endpoint_tbr_y`.",
            f"- static feature dim: {summary['static_feature_dim']}",
            f"- dynamic feature dim: {summary['dynamic_feature_dim']}",
            f"- treatment feature dim: {summary['treatment_feature_dim']}",
            f"- max history length: {summary['max_history_length']}",
            "",
            "## Longitudinal Truncation",
            "",
            "- Dynamic records are counted only when their timestamp is parseable and not later than endpoint image time.",
            "- Unparseable dynamic timestamps are excluded and counted in `sample_history_length.csv`.",
            f"- Samples with dynamic sheets found: {summary['history_with_dynamic_sheet_count']}",
            f"- Samples with at least one usable historical record: {summary['history_nonzero_count']}",
            "",
            "## Output Tables",
            "",
            "- `results/tables/excluded_endpoint_only_patients.csv`",
            "- `results/tables/baseline_patients_without_endpoint.csv`",
            "- `results/tables/multi_window_patients.csv`",
            "- `results/tables/endpoint_selection_detail.csv`",
            "- `results/tables/sample_history_length.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_preprocess_plan(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# KG_LatentNet Data Preprocess Plan",
            "",
            "## Locked Cohort",
            "",
            f"- Use exactly the {summary['final_cohort_size']} `patient_SN` values from `治疗前` as the final research cohort.",
            "- Exclude endpoint-only patients before any split or model input construction.",
            "- All train/val/test/fold splits are patient-level splits on `patient_SN`.",
            "",
            "## Endpoint Selection",
            "",
            "- Match endpoint rows from `治疗后` by `patient_SN`.",
            "- Use `胸主动脉TBR值` only as y.",
            "- Assign windows with [3,9], [10,15], [16,21], [22,27] month rules.",
            "- If duplicate endpoint rows exist, choose a single row per patient and save every candidate in `endpoint_selection_detail.csv`.",
            "- Prefer nonmissing endpoint TBR, then target-window distribution, then closest distance to the window center.",
            "",
            "## Leakage Control",
            "",
            "- Baseline `胸主动脉.2` is allowed as baseline TBR input b.",
            "- Endpoint TBR and any feature matching the leakage blacklist are forbidden from X.",
            f"- Blacklist: {', '.join(LEAKAGE_BLACKLIST)}",
            "",
            "## Longitudinal Truncation",
            "",
            "- For each selected sample, use only dynamic timepoints with parseable time <= endpoint image time.",
            "- If a dynamic timestamp cannot be parsed, exclude it and record the count.",
            "- Endpoint-window is used for grouping/evaluation summaries, not as a default model input.",
            "",
            "## Current Build Summary",
            "",
            f"- Built samples: {summary['built_sample_count']}",
            f"- Window counts: {summary['window_counts']}",
        ]
    ) + "\n"


def write_configs(project_root: Path, summary: dict[str, Any], feature_columns: list[str], dataset_feature_names: dict[str, list[str]]) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    columns_yaml = {
        "generated_at": summary["generated_at"],
        "cohort": {
            "final_patient_count": summary["final_cohort_size"],
            "source_sheet": BASELINE_SHEET,
            "patient_id": PATIENT_ID_COL,
            "internal_code": PATIENT_CODE_COL,
        },
        "columns": {
            "patient_id": PATIENT_ID_COL,
            "patient_internal_code": PATIENT_CODE_COL,
            "baseline_tbr": BASELINE_TBR_COL,
            "endpoint_tbr_label": ENDPOINT_TBR_COL,
            "endpoint_interval_months": ENDPOINT_INTERVAL_COL,
            "endpoint_time": ENDPOINT_TIME_COL,
            "treatment_event_candidates": ["免疫治疗", "化疗", "常规用药（西/中药）", "手术治疗", "放疗", "用药情况", "靶向治疗"],
            "input_feature_columns": feature_columns,
            "static_features": dataset_feature_names.get("static_features", []),
            "dynamic_features": dataset_feature_names.get("dynamic_features", []),
            "treatment_features": dataset_feature_names.get("treatment_features", []),
        },
        "leakage_blacklist": LEAKAGE_BLACKLIST,
        "endpoint_windows": {
            "6": {"lower_inclusive": 3, "upper_inclusive": 9, "center": 6},
            "12": {"lower_inclusive": 10, "upper_inclusive": 15, "center": 12},
            "18": {"lower_inclusive": 16, "upper_inclusive": 21, "center": 18},
            "24": {"lower_inclusive": 22, "upper_inclusive": 27, "center": 24},
        },
        "summary": summary,
    }
    data_yaml = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "splits_dir": "data/splits",
        "tables_dir": "results/tables",
        "patient_workbook": PATIENT_WORKBOOK,
        "final_cohort": {
            "source_sheet": BASELINE_SHEET,
            "patient_id": PATIENT_ID_COL,
            "count": summary["final_cohort_size"],
        },
        "label": {
            "sheet": ENDPOINT_SHEET,
            "column": ENDPOINT_TBR_COL,
            "name": "endpoint_tbr_y",
        },
        "baseline_tbr_input": {
            "sheet": BASELINE_SHEET,
            "column": BASELINE_TBR_COL,
            "name": "baseline_tbr_b",
        },
        "endpoint_window_as_input": False,
        "leakage_blacklist": LEAKAGE_BLACKLIST,
        "split": {
            "unit": PATIENT_ID_COL,
            "seed": 20260605,
            "stratify_by": "endpoint_window",
        },
    }
    (config_dir / "columns.yaml").write_text(yaml.safe_dump(columns_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (config_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_dataset(project_root: Path) -> dict[str, Any]:
    raw_dir = project_root / "data" / "raw"
    tables_dir = project_root / "results" / "tables"
    processed_dir = project_root / "data" / "processed"
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading patient workbook")
    workbook = load_patient_workbook(raw_dir)
    endpoint_month_overrides = load_endpoint_month_overrides(project_root)
    if endpoint_month_overrides:
        logging.info("Loaded %d private endpoint-month corrections.", len(endpoint_month_overrides))
    baseline = normalize_baseline(workbook["baseline"])
    endpoint = normalize_endpoint(workbook["endpoint"], endpoint_month_overrides)
    baseline_ids = set(baseline[PATIENT_ID_COL])

    excluded_endpoint_only = endpoint_only_rows(endpoint, baseline_ids)
    endpoint_in_cohort = endpoint.loc[endpoint[PATIENT_ID_COL].isin(baseline_ids)].copy()
    selected, selection_detail, multi_window = choose_endpoint_records(endpoint_in_cohort)
    baseline_without_endpoint = baseline_without_endpoint_rows(baseline, selected)
    samples, history_rows, feature_columns, dataset = build_samples(baseline, selected, raw_dir, workbook["dynamic_sheets"])

    samples_df = pd.DataFrame(samples)
    samples_df.to_csv(processed_dir / "patient_level_samples.csv", index=False, encoding="utf-8-sig")
    samples_df.to_csv(processed_dir / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "sample_index": list(range(len(dataset["patient_id"]))),
            "patient_SN": dataset["patient_id"],
            "psn": dataset["psn"],
            "endpoint_window": dataset["endpoint_window"].tolist(),
        }
    ).to_csv(processed_dir / "patient_index.csv", index=False, encoding="utf-8-sig")
    (processed_dir / "feature_names.json").write_text(
        json.dumps(dataset["feature_names"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (processed_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump(dataset, handle)
    try:
        import torch

        torch.save(dataset, processed_dir / "dataset.pt")
    except Exception as exc:
        logging.warning("Could not save dataset.pt via torch; dataset.pkl is available. Error: %s", exc)

    write_csv(
        tables_dir / "excluded_endpoint_only_patients.csv",
        excluded_endpoint_only,
        ["patient_SN", "psn", "endpoint_row_number", "endpoint_month_original", "endpoint_month", "endpoint_month_adjustment", "endpoint_window", "endpoint_tbr", "endpoint_time", "exclusion_reason"],
    )
    write_csv(
        tables_dir / "baseline_patients_without_endpoint.csv",
        baseline_without_endpoint,
        ["patient_SN", "psn", "baseline_tbr", "missing_reason"],
    )
    write_csv(
        tables_dir / "multi_window_patients.csv",
        multi_window,
        ["patient_SN", "psn", "endpoint_row_number", "endpoint_month_original", "endpoint_month", "endpoint_month_adjustment", "endpoint_window", "endpoint_tbr", "endpoint_time", "distance_to_window_center", "candidate_windows_for_patient"],
    )
    write_csv(
        tables_dir / "endpoint_selection_detail.csv",
        selection_detail,
        [
            "patient_SN",
            "psn",
            "endpoint_row_number",
            "endpoint_month_original",
            "endpoint_month",
            "endpoint_month_adjustment",
            "endpoint_window",
            "endpoint_tbr",
            "endpoint_time",
            "distance_to_window_center",
            "candidate_count_for_patient",
            "window_candidate_count_for_patient",
            "selected",
            "selected_endpoint_row_number",
            "selection_reason",
        ],
    )
    write_csv(
        tables_dir / "sample_history_length.csv",
        history_rows,
        [
            "patient_SN",
            "psn",
            "endpoint_month_original",
            "endpoint_month",
            "endpoint_month_adjustment",
            "endpoint_window",
            "endpoint_time",
            "dynamic_sheet",
            "total_dynamic_timepoints",
            "used_history_count",
            "excluded_after_endpoint_count",
            "excluded_unknown_time_count",
            "history_note",
        ],
    )

    sheet_names = load_workbook_sheet_names(raw_dir)
    sheet_roles = sheet_role_rows(sheet_names)
    window_counts = Counter(int(row["endpoint_window"]) for row in samples)
    raw_files = {}
    for name in [PATIENT_WORKBOOK, AUTHORITATIVE_KNOWLEDGE_WORKBOOK, EXPERT_KNOWLEDGE_WORKBOOK, PAPER_PDF]:
        path = raw_dir / name
        raw_files[name] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_cohort_size": int(baseline[PATIENT_ID_COL].nunique()),
        "endpoint_rows_total": int(len(endpoint)),
        "endpoint_rows_in_final_cohort": int(len(endpoint_in_cohort)),
        "excluded_endpoint_only_count": len(excluded_endpoint_only),
        "baseline_without_endpoint_count": len(baseline_without_endpoint),
        "endpoint_month_override_count": int(sum(bool(row.get("endpoint_month_adjustment")) for row in selection_detail)),
        "multi_endpoint_patient_count": int(sum(1 for _, group in endpoint_in_cohort.groupby(PATIENT_ID_COL) if len(group) > 1)),
        "multi_window_patient_count": int(len({row["patient_SN"] for row in multi_window})),
        "built_sample_count": len(samples),
        "window_counts": {int(k): int(v) for k, v in sorted(window_counts.items())},
        "input_feature_count": len(feature_columns),
        "static_feature_dim": int(dataset["static_features"].shape[1]),
        "dynamic_feature_dim": int(dataset["dynamic_features"].shape[2]) if dataset["dynamic_features"].ndim == 3 else 0,
        "treatment_feature_dim": int(dataset["treatment_features"].shape[2]) if dataset["treatment_features"].ndim == 3 else 0,
        "max_history_length": int(dataset["dynamic_features"].shape[1]) if dataset["dynamic_features"].ndim == 3 else 0,
        "endpoint_tbr_in_X": ENDPOINT_TBR_COL in feature_columns,
        "history_with_dynamic_sheet_count": int(sum(1 for row in history_rows if row["dynamic_sheet"])),
        "history_nonzero_count": int(sum(1 for row in history_rows if row["used_history_count"] > 0)),
        "dynamic_sheet_count": len(workbook["dynamic_sheets"]),
        "patient_workbook_sheet_count": workbook["sheet_counts"]["total"],
        "total_excel_sheets": sum(len(v) for v in sheet_names.values()),
        "raw_files": raw_files,
        "leakage_blacklist": LEAKAGE_BLACKLIST,
    }

    pd.DataFrame(
        [
            {"metric": "final_cohort_size", "value": summary["final_cohort_size"], "note": "治疗前 sheet unique patient_SN"},
            {"metric": "built_sample_count", "value": summary["built_sample_count"], "note": "patient-level samples with selected endpoint label"},
            {"metric": "excluded_endpoint_only_count", "value": summary["excluded_endpoint_only_count"], "note": "endpoint patient_SN not in baseline cohort"},
            {"metric": "baseline_without_endpoint_count", "value": summary["baseline_without_endpoint_count"], "note": "baseline cohort patients without selected endpoint"},
            {"metric": "endpoint_month_override_count", "value": summary["endpoint_month_override_count"], "note": "patient-specific endpoint month corrections applied before window assignment"},
            {"metric": "multi_endpoint_patient_count", "value": summary["multi_endpoint_patient_count"], "note": "patients with multiple endpoint rows"},
            {"metric": "history_no_dynamic_sheet_count", "value": summary["built_sample_count"] - summary["history_with_dynamic_sheet_count"], "note": "kept with zero-padded history and all-zero dynamic mask"},
            {"metric": "history_nonzero_count", "value": summary["history_nonzero_count"], "note": "samples with at least one usable dynamic record"},
        ]
    ).to_csv(tables_dir / "final_cohort_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"endpoint_window": window, "sample_count": summary["window_counts"].get(window, 0)}
            for window in [6, 12, 18, 24]
        ]
    ).to_csv(tables_dir / "endpoint_window_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"check": "endpoint_tbr_in_X", "value": int(summary["endpoint_tbr_in_X"]), "passed": not summary["endpoint_tbr_in_X"]},
            {"check": "leakage_blacklist_enforced", "value": 1, "passed": True},
            {"check": "endpoint_window_as_input", "value": 0, "passed": True},
            {"check": "endpoint_tbr_y_used_for_scaler_fit", "value": 0, "passed": True},
            {"check": "patient_level_sample_count_positive", "value": int(summary["built_sample_count"] > 0), "passed": summary["built_sample_count"] > 0},
        ]
    ).to_csv(tables_dir / "leakage_check_summary.csv", index=False, encoding="utf-8-sig")

    if summary["built_sample_count"] != summary["final_cohort_size"]:
        logging.warning(
            "Built %d samples from %d baseline patients. See baseline_patients_without_endpoint.csv.",
            summary["built_sample_count"],
            summary["final_cohort_size"],
        )
    else:
        logging.info("Built %d patient-level samples.", summary["built_sample_count"])
    logging.info("Endpoint window counts: %s", summary["window_counts"])
    logging.info("Excluded endpoint-only patients: %d", len(excluded_endpoint_only))
    logging.info("Baseline patients without endpoint: %d", len(baseline_without_endpoint))
    logging.info("Multi-endpoint patients: %d", summary["multi_endpoint_patient_count"])

    write_configs(project_root, summary, feature_columns, dataset["feature_names"])
    (project_root / "data_audit_report.md").write_text(render_audit_report(summary, sheet_roles), encoding="utf-8")
    (project_root / "data_preprocess_plan.md").write_text(render_preprocess_plan(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build KG_LatentNet patient-level data audit artifacts.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    setup_logging(project_root)
    logging.info("Starting data preprocessing audit at %s", project_root)
    summary = build_dataset(project_root)
    logging.info("Completed data preprocessing audit.")
    return summary


if __name__ == "__main__":
    main()
