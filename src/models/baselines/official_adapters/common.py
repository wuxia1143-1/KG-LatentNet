from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


def prepend_paths(*paths: Path) -> None:
    for path in reversed([str(p) for p in paths if p.exists()]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def source_version(project_root: Path, baseline_dir: str, archive_name: str | None = None, observed_commit: str | None = None) -> str:
    repo = project_root / "remote_baselines" / baseline_dir
    git_dir = repo / ".git"
    if git_dir.exists():
        import subprocess

        try:
            return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass
    if archive_name:
        archive = project_root / "remote_baselines" / "_uploaded_archives" / archive_name
        if archive.exists():
            digest = hashlib.sha256()
            with archive.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            prefix = f"observed_repo_head:{observed_commit};" if observed_commit else ""
            return f"{prefix}uploaded_zip_sha256:{digest.hexdigest()}"
    return ""


def observed_sequence_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (batch["dynamic_mask"].sum(dim=-1) > 0) | (batch["treatment_features"].abs().sum(dim=-1) > 0)


def safe_lengths(mask: torch.Tensor) -> torch.Tensor:
    return mask.long().sum(dim=1).clamp_min(1)


def add_dummy_observation_for_empty_history(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_values = values.clone()
    safe_mask = mask.clone()
    empty = safe_mask.sum(dim=(1, 2)) <= 0
    if bool(empty.any()):
        safe_mask[empty, 0, 0] = 1.0
        safe_values[empty, 0, 0] = 0.0
    return safe_values, safe_mask


class OfficialAdapterBase(nn.Module):
    baseline_name: str = ""
    official_repo_url: str = ""
    official_model_class_used: str = ""
    official_entry_script: str = ""
    official_dependencies_file: str = ""
    adapter_only_for_data_mapping: bool = True
    official_model_forward_used: bool = True
    modified_official_code: bool = False
    patch_file: str = ""
    license_name: str = ""

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = Path(project_root)

    @property
    def adapter_file(self) -> str:
        return type(self).__module__.replace(".", "/") + ".py"

    def input_usage(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "patient_id": True,
            "static_features": True,
            "dynamic_features": True,
            "dynamic_mask": True,
            "delta_time": True,
            "treatment_features": True,
            "baseline_tbr_b": True,
            "endpoint_tbr_y": "loss_only",
            "endpoint_window": "metadata_only",
            "endpoint_time": "metadata_only",
            "prior_matrix": False,
            "usage_note": "Official model core is used; adapter maps KG_LatentNet tensors to the official input interface.",
        }
