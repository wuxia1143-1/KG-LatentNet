from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PATIENT_WORKBOOK = os.environ.get("KG_LATENTNET_PATIENT_WORKBOOK", "patient_workbook.xlsx")
AUTHORITATIVE_KNOWLEDGE_WORKBOOK = os.environ.get(
    "KG_LATENTNET_KNOWLEDGE_WORKBOOK",
    "medical_prior_knowledge.xlsx",
)
EXPERT_KNOWLEDGE_WORKBOOK = os.environ.get(
    "KG_LATENTNET_EXPERT_KNOWLEDGE_WORKBOOK",
    "medical_prior_knowledge_expert.xlsx",
)
PAPER_PDF = os.environ.get("KG_LATENTNET_PAPER_PDF", "KG_LatentNet_paper.pdf")


def project_root_from_file(file: str | Path) -> Path:
    return Path(file).resolve().parents[2]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return normalize_text(value).lower() in {"", "na", "n/a", "nan", "none", "null", "-", "--", "无"}
    return False


def dedupe_columns(columns: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    output: list[str] = []
    for idx, name in enumerate(columns, start=1):
        clean = normalize_text(name) or f"unnamed_{idx}"
        counts[clean] += 1
        if counts[clean] > 1:
            clean = f"{clean}.{counts[clean]}"
        output.append(clean)
    return output


def header_from_rows(raw: pd.DataFrame, header_rows: int = 2) -> list[str]:
    if raw.empty:
        return []
    row1 = raw.iloc[0].tolist() if len(raw) >= 1 else []
    row2 = raw.iloc[1].tolist() if len(raw) >= 2 and header_rows >= 2 else []
    width = raw.shape[1]
    headers: list[str] = []
    for idx in range(width):
        first = normalize_text(row1[idx] if idx < len(row1) else "")
        second = normalize_text(row2[idx] if idx < len(row2) else "")
        headers.append(second or first or f"unnamed_{idx + 1}")
    return dedupe_columns(headers)


def dataframe_from_header_rows(raw: pd.DataFrame, header_rows: int = 2) -> pd.DataFrame:
    raw = raw.where(pd.notna(raw), None)
    headers = header_from_rows(raw, header_rows=header_rows)
    body = raw.iloc[header_rows:].copy()
    body.columns = headers
    body = body.loc[:, [not str(col).startswith("unnamed_") or not body[col].map(is_missing).all() for col in body.columns]]
    body = body.loc[~body.apply(lambda row: all(is_missing(value) for value in row), axis=1)]
    return body.reset_index(drop=True)


def load_patient_workbook(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / PATIENT_WORKBOOK
    excel = pd.ExcelFile(path)
    baseline_raw = pd.read_excel(path, sheet_name="治疗前", header=None, dtype=object)
    endpoint_raw = pd.read_excel(path, sheet_name="治疗后", header=None, dtype=object)
    baseline = dataframe_from_header_rows(baseline_raw, header_rows=2)
    endpoint = dataframe_from_header_rows(endpoint_raw, header_rows=2)
    dynamic_sheets = [
        name
        for name in excel.sheet_names
        if str(name).startswith("治疗中") or str(name).isdigit()
    ]
    return {
        "path": path,
        "sheet_names": excel.sheet_names,
        "baseline": baseline,
        "endpoint": endpoint,
        "dynamic_sheets": dynamic_sheets,
        "sheet_counts": {
            "total": len(excel.sheet_names),
            "note": int("备注" in excel.sheet_names),
            "baseline": int("治疗前" in excel.sheet_names),
            "endpoint": int("治疗后" in excel.sheet_names),
            "dynamic": len(dynamic_sheets),
        },
    }


def read_dynamic_sheet(raw_dir: Path, sheet_name: str) -> pd.DataFrame:
    path = raw_dir / PATIENT_WORKBOOK
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
    return raw.where(pd.notna(raw), None)


def load_workbook_sheet_names(raw_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for file_name in [PATIENT_WORKBOOK, AUTHORITATIVE_KNOWLEDGE_WORKBOOK, EXPERT_KNOWLEDGE_WORKBOOK]:
        path = raw_dir / file_name
        result[file_name] = pd.ExcelFile(path).sheet_names
    return result
