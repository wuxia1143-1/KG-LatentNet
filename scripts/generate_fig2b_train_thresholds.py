from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/root/KG_LatentNet_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold
from src.data.prior_alignment import build_aligned_prior_matrix
from src.training.full_5fold_evaluation import extract_kg_latent_outputs
from src.training.validation_tuning import TensorFoldDataset, build_torch_model, collate


root = PROJECT_ROOT
tables = root / "results" / "tables" / "full_5fold"
figs = root / "results" / "figures" / "full_5fold"
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(root)
cfg = yaml.safe_load((root / "configs" / "locked_full_5fold_config.yaml").read_text(encoding="utf-8"))
prior_np, _, prior_checks = build_aligned_prior_matrix(root, dataset["feature_names"]["dynamic_features"])
if not all(bool(row["passed"]) for row in prior_checks):
    raise RuntimeError("prior alignment failed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior = torch.tensor(prior_np, dtype=torch.float32, device=device)
static_dim = int(dataset["static_features"].shape[1])
dynamic_dim = int(dataset["dynamic_features"].shape[2])
treatment_dim = int(dataset["treatment_features"].shape[2])

threshold_rows = []
category_rows = []
for fold in range(5):
    fold_payload = load_fold(root, fold)
    preprocess = pickle.load((root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb"))
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["train_patient_ids"]))
    test_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["test_patient_ids"]))
    train_loader = DataLoader(TensorFoldDataset(train_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(TensorFoldDataset(test_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    params = dict(cfg["models"]["kg_latentnet"]["folds"][f"fold_{fold}"]["selected_params"])
    model = build_torch_model("kg_latentnet", root, static_dim, dynamic_dim, treatment_dim, params).to(device)
    ckpt = torch.load(root / "results" / "checkpoints" / "full_5fold" / f"kg_latentnet_fold{fold}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    train_latent = extract_kg_latent_outputs(model, train_loader, device, prior, fold)["patient_latent_rows"]
    test_latent = extract_kg_latent_outputs(model, test_loader, device, prior, fold)["patient_latent_rows"]
    train_scores = np.asarray([row["latent_state_score"] for row in train_latent], dtype=float)
    q33, q66 = np.quantile(train_scores, [0.33, 0.66])
    threshold_rows.append({"fold": fold, "train_latent_q33": q33, "train_latent_q66": q66, "n_train": len(train_scores), "threshold_source": "train_split_only"})
    for row in test_latent:
        score = float(row["latent_state_score"])
        if score <= q33:
            category = "low"
        elif score <= q66:
            category = "middle"
        else:
            category = "high"
        category_rows.append({
            "patient_id": row["patient_id"],
            "fold": fold,
            "endpoint_window": row["endpoint_window"],
            "latent_state_score": score,
            "latent_category": category,
            "threshold_source": "train_split_q33_q66",
        })

threshold_df = pd.DataFrame(threshold_rows)
category_df = pd.DataFrame(category_rows)
threshold_df.to_csv(tables / "latent_category_thresholds_train_only.csv", index=False, encoding="utf-8-sig")
category_df.to_csv(tables / "latent_category_patient_assignments.csv", index=False, encoding="utf-8-sig")
dist = category_df.groupby(["endpoint_window", "latent_category"]).size().reset_index(name="n")
total = category_df.groupby("endpoint_window").size().reset_index(name="total")
dist = dist.merge(total, on="endpoint_window")
dist["proportion"] = dist["n"] / dist["total"]
dist.to_csv(tables / "latent_category_distribution.csv", index=False, encoding="utf-8-sig")

pivot = dist.pivot(index="endpoint_window", columns="latent_category", values="proportion").fillna(0)
for col in ["low", "middle", "high"]:
    if col not in pivot:
        pivot[col] = 0.0
pivot = pivot[["low", "middle", "high"]].sort_index()
plt.figure(figsize=(6, 4))
bottom = np.zeros(len(pivot))
colors = {"low": "#4c78a8", "middle": "#f58518", "high": "#54a24b"}
for col in ["low", "middle", "high"]:
    plt.bar([str(int(x)) for x in pivot.index], pivot[col], bottom=bottom, label=col, color=colors[col])
    bottom += pivot[col].to_numpy()
plt.ylim(0, 1)
plt.xlabel("Follow-up month")
plt.ylabel("Proportion")
plt.title("Latent state categories by follow-up")
plt.legend(title="Train-threshold category")
plt.tight_layout()
plt.savefig(figs / "fig2b_latent_state_categories.png", dpi=200)
plt.close()
print("generated fig2b and train-only latent category tables")
