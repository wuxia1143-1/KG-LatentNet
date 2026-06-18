from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/root/KG_LatentNet_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold
from src.data.prior_alignment import build_aligned_prior_matrix
from src.models.kg_latentnet_v2 import KGLatentNetV2
from src.training.train_kg_latentnet_v2 import (
    ResidualTensorDataset, collate, compute_metrics, build_prior_matrix, write_csv,
)


def load_locked_config(project_root: Path) -> dict:
    import yaml
    path = project_root / "configs" / "locked_kg_v2_fusion_fix_config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_fold_dataset(dataset, fold_payload, preprocess, fold_idx):
    train_ids = fold_payload["train_patient_ids"]
    val_ids = fold_payload["val_patient_ids"]
    train_indices = ids_to_indices(dataset, train_ids)
    val_indices = ids_to_indices(dataset, val_ids)
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    n_train = len(train_arrays["patient_id"])
    n_val = len(val_arrays["patient_id"])

    train_win = train_arrays["endpoint_window"]
    val_win = val_arrays["endpoint_window"]

    train_base = train_arrays["baseline_tbr_b"]
    val_base = val_arrays["baseline_tbr_b"]
    train_y = train_arrays["endpoint_tbr_y"]
    val_y = val_arrays["endpoint_tbr_y"]
    train_delta = train_y - train_base
    val_delta = val_y - val_base

    train_dyn = train_arrays["dynamic_features"]
    val_dyn = val_arrays["dynamic_features"]
    train_mask = train_arrays["dynamic_mask"]
    val_mask = val_arrays["dynamic_mask"]

    train_no_dyn = int(np.sum(train_dyn.reshape(n_train, -1).sum(axis=1) == 0))
    val_no_dyn = int(np.sum(val_dyn.reshape(n_val, -1).sum(axis=1) == 0))

    if train_mask.ndim == 3:
        train_obs_t = (train_mask.sum(axis=-1) > 0).sum(axis=1)
        val_obs_t = (val_mask.sum(axis=-1) > 0).sum(axis=1)
    else:
        train_obs_t = (train_mask > 0).sum(axis=1)
        val_obs_t = (val_mask > 0).sum(axis=1)
    T = train_mask.shape[1]
    train_low_obs = int(np.sum(train_obs_t < T * 0.3))
    val_low_obs = int(np.sum(val_obs_t < T * 0.3))

    stats = {
        "fold": fold_idx,
        "n_train": n_train,
        "n_val": n_val,
        "train_baseline_mean": round(float(np.mean(train_base)), 6),
        "train_baseline_std": round(float(np.std(train_base)), 6),
        "val_baseline_mean": round(float(np.mean(val_base)), 6),
        "val_baseline_std": round(float(np.std(val_base)), 6),
        "train_delta_mean": round(float(np.mean(train_delta)), 6),
        "train_delta_std": round(float(np.std(train_delta)), 6),
        "val_delta_mean": round(float(np.mean(val_delta)), 6),
        "val_delta_std": round(float(np.std(val_delta)), 6),
        "train_no_dynamic": train_no_dyn,
        "val_no_dynamic": val_no_dyn,
        "train_low_obs_patients": train_low_obs,
        "val_low_obs_patients": val_low_obs,
        "train_obs_mean": round(float(train_obs_t.mean()), 1),
        "val_obs_mean": round(float(val_obs_t.mean()), 1),
    }
    for w in [6, 12, 18, 24]:
        stats[f"train_win_{w}"] = int(np.sum(train_win == w))
        stats[f"val_win_{w}"] = int(np.sum(val_win == w))

    return stats, train_arrays, val_arrays


def train_fold_and_collect(project_root, fold, fold_config, device, dataset, preprocess, fold_payload):
    seed = int(fold_config.get("seed", 20260606))
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)

    hidden_dim = int(fold_config["hidden_dim"])
    latent_dim = int(fold_config["latent_dim"])
    summary_dim = int(fold_config["summary_dim"])
    dropout = float(fold_config["dropout"])
    lr = float(fold_config["learning_rate"])
    wd = float(fold_config["weight_decay"])
    loss_type = str(fold_config.get("loss", "Huber"))
    huber_delta = float(fold_config["huber_delta"])
    model_variant = str(fold_config["model_variant"])
    readout_head = str(fold_config.get("readout_head", "shared"))
    lambda_delta = float(fold_config["lambda_delta"])
    lambda_anchor = float(fold_config["lambda_anchor"])
    lambda_prior = float(fold_config.get("lambda_prior", 0.0))
    lambda_smooth = float(fold_config.get("lambda_smooth", 0.0))
    lambda_disentangle = float(fold_config.get("lambda_disentangle", 0.0))
    gradient_clip = float(fold_config["gradient_clip"])
    patience = int(fold_config["early_stopping_patience"])
    batch_size = 16
    epochs = 200

    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    y_train = train_arrays["endpoint_tbr_y"]
    y_range = (float(np.nanmin(y_train)), float(np.nanmax(y_train)))

    train_loader = DataLoader(
        ResidualTensorDataset(train_arrays), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        ResidualTensorDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate
    )

    static_dim = dataset["static_features"].shape[1]
    dynamic_dim = dataset["dynamic_features"].shape[2]
    treatment_dim = dataset["treatment_features"].shape[2]

    model = KGLatentNetV2(
        static_dim=static_dim, dynamic_dim=dynamic_dim, treatment_dim=treatment_dim,
        hidden_dim=hidden_dim, latent_dim=latent_dim, summary_dim=summary_dim,
        dropout=dropout, model_variant=model_variant, readout_head=readout_head,
        huber_delta=huber_delta,
    ).to(device)

    prior_matrix = build_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"]).to(device)

    if loss_type == "Huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch["tensors"].items()}
            outputs = model(tensor_batch, prior_matrix=prior_matrix)
            loss_dict = KGLatentNetV2.compute_loss(
                outputs, tensor_batch, criterion,
                lambda_delta=lambda_delta, lambda_anchor=lambda_anchor,
                lambda_prior=lambda_prior, lambda_smooth=lambda_smooth,
                lambda_disentangle=lambda_disentangle, y_range=y_range,
            )
            loss = loss_dict["total"]
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        model.eval()
        val_rows = []
        with torch.no_grad():
            for batch in val_loader:
                tensor_batch = {k: v.to(device) for k, v in batch["tensors"].items()}
                outputs = model(tensor_batch, prior_matrix=prior_matrix)
                y_pred = outputs["y_pred"].cpu().numpy()
                y_true = tensor_batch["endpoint_tbr_y"].squeeze(-1).cpu().numpy()
                baseline = tensor_batch["baseline_tbr_b"].squeeze(-1).cpu().numpy()
                for i in range(len(y_pred)):
                    val_rows.append(float(abs(y_pred[i] - y_true[i])))
        val_mae = float(np.mean(val_rows))
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    elapsed = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    def collect_preds(loader):
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in loader:
                tensor_batch = {k: v.to(device) for k, v in batch["tensors"].items()}
                outputs = model(tensor_batch, prior_matrix=prior_matrix)
                y_pred = outputs["y_pred"].cpu().numpy()
                delta_pred = outputs["delta_pred"].cpu().numpy()
                delta_latent = outputs["delta_latent"].cpu().numpy()
                delta_summary = outputs["delta_summary"].cpu().numpy()
                delta_treatment = outputs["delta_treatment"].cpu().numpy()
                gate_w = outputs["gate_weights"].cpu().numpy()
                y_true = tensor_batch["endpoint_tbr_y"].squeeze(-1).cpu().numpy()
                baseline = tensor_batch["baseline_tbr_b"].squeeze(-1).cpu().numpy()
                delta_true = y_true - baseline
                for i in range(len(y_pred)):
                    rows.append({
                        "patient_id": batch["patient_id"][i],
                        "fold": fold,
                        "endpoint_window": int(batch["endpoint_window"][i].item()),
                        "baseline_tbr_b": float(baseline[i]),
                        "y_true": float(y_true[i]),
                        "delta_true": float(delta_true[i]),
                        "delta_pred": float(delta_pred[i]),
                        "delta_latent": float(delta_latent[i]),
                        "delta_summary": float(delta_summary[i]),
                        "delta_treatment": float(delta_treatment[i]),
                        "gate_latent": float(gate_w[i, 0]),
                        "gate_summary": float(gate_w[i, 1]),
                        "gate_treatment": float(gate_w[i, 2]),
                        "y_pred": float(y_pred[i]),
                        "absolute_error": float(abs(y_pred[i] - y_true[i])),
                    })
        return rows

    train_preds = collect_preds(train_loader)
    val_preds = collect_preds(val_loader)

    yt_tr = np.array([r["y_true"] for r in train_preds])
    yp_tr = np.array([r["y_pred"] for r in train_preds])
    yt_va = np.array([r["y_true"] for r in val_preds])
    yp_va = np.array([r["y_pred"] for r in val_preds])

    train_metrics = compute_metrics(yt_tr, yp_tr)
    val_metrics = compute_metrics(yt_va, yp_va)

    gate_lats = np.array([r["gate_latent"] for r in val_preds])
    gate_sums = np.array([r["gate_summary"] for r in val_preds])
    gate_trts = np.array([r["gate_treatment"] for r in val_preds])

    return {
        "train_preds": train_preds,
        "val_preds": val_preds,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "runtime_sec": elapsed,
        "gate_mean": {
            "latent": round(float(gate_lats.mean()), 6),
            "summary": round(float(gate_sums.mean()), 6),
            "treatment": round(float(gate_trts.mean()), 6),
        },
        "gate_std": {
            "latent": round(float(gate_lats.std()), 6),
            "summary": round(float(gate_sums.std()), 6),
            "treatment": round(float(gate_trts.std()), 6),
        },
        "gate_max_single": round(float(np.maximum(np.maximum(gate_lats, gate_sums), gate_trts).mean()), 6),
    }


def generate_figures(fold_results, fold_stats, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.size"] = 10
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")
        return

    fig_dir = output_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    folds = sorted(fold_results.keys())
    train_maes = [fold_results[f]["train_metrics"]["mae"] for f in folds]
    val_maes = [fold_results[f]["val_metrics"]["mae"] for f in folds]
    x = np.arange(len(folds))
    w = 0.35
    axes[0].bar(x - w/2, train_maes, w, label="Train MAE", color="#2196F3")
    axes[0].bar(x + w/2, val_maes, w, label="Val MAE", color="#F44336")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Train vs Val MAE (Fold 1 & 3 Focus)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {f}" for f in folds])
    axes[0].legend()
    axes[0].axhline(y=0.2317, color="green", linestyle="--", alpha=0.7, label="RF target")
    for i, f in enumerate(folds):
        gap = val_maes[i] - train_maes[i]
        axes[0].annotate(f"gap={gap:.3f}", (x[i], val_maes[i] + 0.01), ha="center", fontsize=8)

    gaps = [fold_results[f]["val_metrics"]["mae"] - fold_results[f]["train_metrics"]["mae"] for f in folds]
    colors = ["#F44336" if g > 0.15 else "#FF9800" if g > 0.05 else "#4CAF50" for g in gaps]
    axes[1].bar(x, gaps, color=colors)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Val MAE - Train MAE")
    axes[1].set_title("Overfitting Gap (Train-Val)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {f}" for f in folds])
    axes[1].axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Severe overfit threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "kg_v2_fold13_train_val_gap.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, fold in enumerate(folds):
        vp = fold_results[fold]["val_preds"]
        dt = np.array([r["delta_true"] for r in vp])
        dp = np.array([r["delta_pred"] for r in vp])
        axes[idx].scatter(dt, dp, alpha=0.5, s=20, c="#2196F3")
        lims = [min(dt.min(), dp.min()) - 0.1, max(dt.max(), dp.max()) + 0.1]
        axes[idx].plot(lims, lims, "r--", alpha=0.5)
        axes[idx].set_xlabel("delta_true")
        axes[idx].set_ylabel("delta_pred")
        corr = np.corrcoef(dt, dp)[0, 1] if len(dt) > 1 else 0
        mae = np.mean(np.abs(dt - dp))
        axes[idx].set_title(f"Fold {fold}: delta_true vs pred (corr={corr:.3f}, MAE={mae:.3f})")
        axes[idx].set_xlim(lims)
        axes[idx].set_ylim(lims)
    plt.tight_layout()
    plt.savefig(fig_dir / "kg_v2_fold13_delta_true_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, fold in enumerate(folds):
        vp = fold_results[fold]["val_preds"]
        gl = np.array([r["gate_latent"] for r in vp])
        gs = np.array([r["gate_summary"] for r in vp])
        gt = np.array([r["gate_treatment"] for r in vp])
        data = [gl, gs, gt]
        bp = axes[idx].boxplot(data, labels=["Latent", "Summary", "Treatment"], patch_artist=True)
        colors_box = ["#2196F3", "#4CAF50", "#FF9800"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[idx].set_ylabel("Gate Weight")
        axes[idx].set_title(f"Fold {fold}: Gate Weight Distribution")
        axes[idx].set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "kg_v2_fold13_gate_weight_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, fold in enumerate(folds):
        vp = fold_results[fold]["val_preds"]
        base = np.array([r["baseline_tbr_b"] for r in vp])
        ae = np.array([r["absolute_error"] for r in vp])
        axes[idx].scatter(base, ae, alpha=0.5, s=20, c="#F44336")
        z = np.polyfit(base, ae, 1)
        p = np.poly1d(z)
        xs = np.linspace(base.min(), base.max(), 100)
        axes[idx].plot(xs, p(xs), "b--", alpha=0.5)
        axes[idx].set_xlabel("baseline_tbr_b")
        axes[idx].set_ylabel("Absolute Error")
        corr = np.corrcoef(base, ae)[0, 1] if len(base) > 1 else 0
        axes[idx].set_title(f"Fold {fold}: Error vs Baseline (corr={corr:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "kg_v2_fold13_error_by_baseline_tbr.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, fold in enumerate(folds):
        vp = fold_results[fold]["val_preds"]
        windows = np.array([r["endpoint_window"] for r in vp])
        ae = np.array([r["absolute_error"] for r in vp])
        ws = [6, 12, 18, 24]
        mean_ae = []
        counts = []
        for w in ws:
            mask = windows == w
            if mask.any():
                mean_ae.append(float(np.mean(ae[mask])))
                counts.append(int(mask.sum()))
            else:
                mean_ae.append(0)
                counts.append(0)
        bars = axes[idx].bar([str(w) for w in ws], mean_ae, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
        for bar, cnt in zip(bars, counts):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f"n={cnt}", ha="center", fontsize=8)
        axes[idx].set_xlabel("Endpoint Window (months)")
        axes[idx].set_ylabel("Mean Absolute Error")
        axes[idx].set_title(f"Fold {fold}: MAE by Window")
    plt.tight_layout()
    plt.savefig(fig_dir / "kg_v2_fold13_error_by_window.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    print(f"Project root: {project_root}")

    print("Loading dataset...")
    dataset = load_dataset(project_root)
    print(f"Dataset loaded: {len(dataset['patient_id'])} patients")

    print("Loading locked config...")
    config = load_locked_config(project_root)
    folds_config = config["folds"]

    print("\n=== Dataset-Level Analysis (All 5 Folds) ===")
    all_fold_stats = []
    for fold in range(5):
        fold_payload = load_fold(project_root, fold)
        with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as f:
            preprocess = pickle.load(f)
        stats, _, _ = analyze_fold_dataset(dataset, fold_payload, preprocess, fold)
        all_fold_stats.append(stats)
        print(f"Fold {fold}: train={stats['n_train']} val={stats['n_val']} "
              f"no_dyn_train={stats['train_no_dynamic']} no_dyn_val={stats['val_no_dynamic']} "
              f"low_obs_train={stats['train_low_obs_patients']} low_obs_val={stats['val_low_obs_patients']}")

    diag_csv_path = project_root / "results" / "tables" / "full_5fold" / "kg_v2_fold13_failure_diagnosis.csv"
    diag_csv_path.parent.mkdir(parents=True, exist_ok=True)
    diag_fields = list(all_fold_stats[0].keys())
    with diag_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fields)
        writer.writeheader()
        for row in all_fold_stats:
            writer.writerow(row)

    print(f"\nDataset analysis saved to {diag_csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    fold_results = {}
    for fold in [1, 3]:
        print(f"\n=== Training Fold {fold} (from locked config) ===")
        fold_config = folds_config[str(fold)]
        print(f"  variant={fold_config['model_variant']} hidden={fold_config['hidden_dim']} "
              f"latent={fold_config['latent_dim']} summary={fold_config['summary_dim']} "
              f"dropout={fold_config['dropout']} lr={fold_config['learning_rate']} "
              f"wd={fold_config['weight_decay']} lambda_anchor={fold_config['lambda_anchor']}")

        fold_payload = load_fold(project_root, fold)
        with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as f:
            preprocess = pickle.load(f)

        result = train_fold_and_collect(
            project_root, fold, fold_config, device, dataset, preprocess, fold_payload,
        )
        fold_results[fold] = result

        tm = result["train_metrics"]
        vm = result["val_metrics"]
        gap = vm["mae"] - tm["mae"]
        print(f"  Train MAE={tm['mae']:.6f} RMSE={tm['rmse']:.6f} R2={tm['r2']:.6f}")
        print(f"  Val   MAE={vm['mae']:.6f} RMSE={vm['rmse']:.6f} R2={vm['r2']:.6f}")
        print(f"  Gap = {gap:.6f}")
        print(f"  Best epoch = {result['best_epoch']}")
        print(f"  Gate mean: L={result['gate_mean']['latent']:.4f} S={result['gate_mean']['summary']:.4f} T={result['gate_mean']['treatment']:.4f}")
        print(f"  Gate std:  L={result['gate_std']['latent']:.4f} S={result['gate_std']['summary']:.4f} T={result['gate_std']['treatment']:.4f}")
        print(f"  Gate max single (avg) = {result['gate_max_single']:.4f}")

        pred_fields = [
            "patient_id", "fold", "endpoint_window", "baseline_tbr_b",
            "y_true", "delta_true", "delta_pred", "delta_latent", "delta_summary", "delta_treatment",
            "gate_latent", "gate_summary", "gate_treatment", "y_pred", "absolute_error",
        ]
        train_pred_path = project_root / "results" / "predictions" / "full_5fold" / f"kg_v2_fold{fold}_train_predictions.csv"
        val_pred_path = project_root / "results" / "predictions" / "full_5fold" / f"kg_v2_fold{fold}_val_predictions.csv"
        write_csv(train_pred_path, result["train_preds"], pred_fields)
        write_csv(val_pred_path, result["val_preds"], pred_fields)
        print(f"  Predictions saved: {train_pred_path.name}, {val_pred_path.name}")

    print("\n=== Fold 1 vs Fold 3 Comparison ===")
    for fold in [1, 3]:
        r = fold_results[fold]
        print(f"Fold {fold}:")
        print(f"  variant={folds_config[str(fold)]['model_variant']}")
        print(f"  train_mae={r['train_metrics']['mae']:.6f} val_mae={r['val_metrics']['mae']:.6f} gap={r['val_metrics']['mae'] - r['train_metrics']['mae']:.6f}")
        print(f"  best_epoch={r['best_epoch']} runtime={r['runtime_sec']:.1f}s")

    print("\n=== Generating Figures ===")
    fig_dir = project_root / "results" / "figures" / "full_5fold"
    generate_figures(fold_results, all_fold_stats, fig_dir)

    print("\n=== Diagnosis Complete ===")
    print("Key findings:")
    for fold in [1, 3]:
        r = fold_results[fold]
        gap = r["val_metrics"]["mae"] - r["train_metrics"]["mae"]
        overfit = "SEVERE" if gap > 0.2 else "MODERATE" if gap > 0.1 else "MILD"
        print(f"  Fold {fold}: gap={gap:.4f} ({overfit} overfitting)")
        print(f"    Gate collapse: max_single={r['gate_max_single']:.4f} ({'YES' if r['gate_max_single'] > 0.7 else 'NO'})")


if __name__ == "__main__":
    main()
