from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.load_excel import AUTHORITATIVE_KNOWLEDGE_WORKBOOK, EXPERT_KNOWLEDGE_WORKBOOK


TREATMENT_SHEET_TOKEN = "治疗方案整体影响"
BIOMARKER_SHEET_TOKEN = "变量"

SYNONYM_MAPPING = {
    "血糖": "葡萄糖",
    "总胆固醇": "胆固醇",
    "LDL-C": "低密度脂蛋白胆固醇",
    "HDL-C": "高密度脂蛋白胆固醇",
    "收缩压": "血压",
    "舒张压": "血压",
    "D-二聚体": "D-二聚体",
    "血小板": "血小板",
    "中性粒细胞": "中性粒细胞",
    "淋巴细胞": "淋巴细胞",
    "NLR（中性粒细胞/淋巴细胞）": "NLR（中性粒细胞/淋巴细胞）",
    "神经元特异性烯醇化酶": "神经元特异性烯醇化酶",
    "鳞状细胞癌相关抗原": "鳞状细胞癌相关抗原",
    "胃泌素释放肽前体": "胃泌素释放肽前体",
    "细胞角蛋白19片段": "细胞角蛋白19片段",
    "高敏肌钙蛋白T": "高敏肌钙蛋白T",
    "肾小球滤过率": "肾小球滤过率",
    "肌酐": "肌酐",
    "脂蛋白a": "脂蛋白a",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()


def direction_value(value: Any) -> float | None:
    text = normalize_text(value)
    if not text or text in {"/", "nan", "NaN"}:
        return None
    if text in {"0", "—", "–", "-", "无"}:
        return 0.0
    score = 0.0
    score += text.count("↑") or text.count("+")
    score -= text.count("↓") or text.count("-")
    if score != 0:
        return float(max(-3.0, min(3.0, score)))
    if "升高" in text or "增加" in text:
        return 1.0
    if "降低" in text or "减少" in text:
        return -1.0
    return None


def first_matching_sheet(path: Path, token: str) -> str:
    sheets = pd.ExcelFile(path).sheet_names
    for sheet in sheets:
        if token in sheet:
            return sheet
    raise ValueError(f"No sheet containing {token!r} in {path}")


def map_variable(name: str, dataset_variables: list[str]) -> str:
    clean = normalize_text(name)
    if clean in dataset_variables:
        return clean
    if clean in SYNONYM_MAPPING:
        mapped = SYNONYM_MAPPING[clean]
        if mapped in dataset_variables:
            return mapped
        for variable in dataset_variables:
            if mapped in variable or variable in mapped:
                return variable
        return mapped
    for variable in dataset_variables:
        if clean and (clean in variable or variable in clean):
            return variable
    return clean


def read_header_table(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype=object)


def combine_values(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    value = float(np.mean(clean))
    return float(max(-1.0, min(1.0, value / 3.0)))


def build_knowledge_prior(project_root: Path) -> dict[str, Any]:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    tables_dir = project_root / "results" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    feature_names = yaml.safe_load((project_root / "configs" / "columns.yaml").read_text(encoding="utf-8"))
    dataset_variables = list(
        dict.fromkeys(
            feature_names["columns"].get("dynamic_features", [])
            + feature_names["columns"].get("input_feature_columns", [])
            + feature_names["columns"].get("treatment_event_candidates", [])
        )
    )
    if not dataset_variables:
        dataset_variables = feature_names["columns"].get("input_feature_columns", [])

    workbooks = [raw_dir / AUTHORITATIVE_KNOWLEDGE_WORKBOOK, raw_dir / EXPERT_KNOWLEDGE_WORKBOOK]
    treatment_rows: dict[tuple[str, str], list[float | None]] = {}
    biomarker_rows: dict[tuple[str, str], list[float | None]] = {}
    mapping_rows: list[dict[str, Any]] = []

    for workbook in workbooks:
        treatment_sheet = first_matching_sheet(workbook, TREATMENT_SHEET_TOKEN)
        treatment_df = read_header_table(workbook, treatment_sheet)
        treatment_col = treatment_df.columns[0]
        for _, row in treatment_df.iterrows():
            treatment = normalize_text(row[treatment_col])
            if not treatment or treatment.startswith("备注"):
                continue
            for column in treatment_df.columns[1:]:
                variable = normalize_text(column)
                mapped = map_variable(variable, dataset_variables)
                value = direction_value(row[column])
                treatment_rows.setdefault((treatment, mapped), []).append(value)
                mapping_rows.append(
                    {
                        "source_file": workbook.name,
                        "source_sheet": treatment_sheet,
                        "source_variable": variable,
                        "mapped_variable": mapped,
                        "mapping_type": "treatment_biomarker",
                    }
                )

        biomarker_sheet = first_matching_sheet(workbook, "变量")
        biomarker_df = read_header_table(workbook, biomarker_sheet)
        row_col = biomarker_df.columns[0]
        for _, row in biomarker_df.iterrows():
            source = map_variable(normalize_text(row[row_col]), dataset_variables)
            if not source:
                continue
            for column in biomarker_df.columns[1:]:
                target = map_variable(normalize_text(column), dataset_variables)
                value = direction_value(row[column])
                biomarker_rows.setdefault((source, target), []).append(value)
                mapping_rows.append(
                    {
                        "source_file": workbook.name,
                        "source_sheet": biomarker_sheet,
                        "source_variable": normalize_text(column),
                        "mapped_variable": target,
                        "mapping_type": "biomarker_biomarker",
                    }
                )

    treatment_names = sorted({key[0] for key in treatment_rows})
    biomarker_names = sorted({key[1] for key in treatment_rows} | {key[0] for key in biomarker_rows} | {key[1] for key in biomarker_rows})
    treatment_index = {name: idx for idx, name in enumerate(treatment_names)}
    biomarker_index = {name: idx for idx, name in enumerate(biomarker_names)}
    treatment_biomarker_prior = np.zeros((len(treatment_names), len(biomarker_names)), dtype=np.float32)
    biomarker_biomarker_prior = np.zeros((len(biomarker_names), len(biomarker_names)), dtype=np.float32)
    for (treatment, biomarker), values in treatment_rows.items():
        treatment_biomarker_prior[treatment_index[treatment], biomarker_index[biomarker]] = combine_values(values)
    for (source, target), values in biomarker_rows.items():
        biomarker_biomarker_prior[biomarker_index[source], biomarker_index[target]] = combine_values(values)

    prior = {
        "treatment_names": treatment_names,
        "biomarker_names": biomarker_names,
        "treatment_biomarker_prior": treatment_biomarker_prior,
        "biomarker_biomarker_prior": biomarker_biomarker_prior,
        "value_range": "[-1, 1]",
        "direction_preserved": True,
    }
    with (processed_dir / "knowledge_prior.pkl").open("wb") as handle:
        pickle.dump(prior, handle)

    with (tables_dir / "knowledge_prior_variable_mapping.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["source_file", "source_sheet", "source_variable", "mapped_variable", "mapping_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)
    pd.DataFrame(
        [
            {"matrix": "treatment_biomarker_prior", "rows": treatment_biomarker_prior.shape[0], "cols": treatment_biomarker_prior.shape[1], "nonzero": int(np.count_nonzero(treatment_biomarker_prior))},
            {"matrix": "biomarker_biomarker_prior", "rows": biomarker_biomarker_prior.shape[0], "cols": biomarker_biomarker_prior.shape[1], "nonzero": int(np.count_nonzero(biomarker_biomarker_prior))},
        ]
    ).to_csv(tables_dir / "knowledge_prior_summary.csv", index=False, encoding="utf-8-sig")

    columns_path = project_root / "configs" / "columns.yaml"
    columns = yaml.safe_load(columns_path.read_text(encoding="utf-8"))
    columns["knowledge_prior_mapping"] = {
        key: map_variable(key, dataset_variables) for key in sorted(SYNONYM_MAPPING)
    }
    columns["knowledge_prior"] = {
        "path": "data/processed/knowledge_prior.pkl",
        "treatment_biomarker_prior_shape": list(treatment_biomarker_prior.shape),
        "biomarker_biomarker_prior_shape": list(biomarker_biomarker_prior.shape),
    }
    columns_path.write_text(yaml.safe_dump(columns, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return prior


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build KG-LatentNet knowledge priors.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    return build_knowledge_prior(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
