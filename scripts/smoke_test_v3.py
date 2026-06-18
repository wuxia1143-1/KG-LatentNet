import sys
sys.path.insert(0, "/root/KG_LatentNet_Project")
from src.models.kg_latentnet_v3 import KGLatentNetV3
print("v3 import OK")
print("variants:", KGLatentNetV3.VARIANTS)

import torch
for variant in KGLatentNetV3.VARIANTS:
    model = KGLatentNetV3(
        static_dim=10, dynamic_dim=31, treatment_dim=11,
        hidden_dim=8, latent_dim=4, summary_dim=16,
        dropout=0.5, model_variant=variant, huber_delta=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {variant}: params={n_params}")
    batch = {
        "static_features": torch.randn(4, 10),
        "dynamic_features": torch.randn(4, 20, 31),
        "dynamic_mask": torch.ones(4, 20, 31),
        "delta_time": torch.randn(4, 20).abs(),
        "treatment_features": torch.randn(4, 20, 11),
        "baseline_tbr_b": torch.randn(4, 1),
        "endpoint_tbr_y": torch.randn(4, 1),
        "endpoint_window": torch.tensor([6, 12, 18, 24]),
    }
    out = model(batch)
    print(f"    y_pred shape={out['y_pred'].shape} gate_entropy={out['gate_entropy'].mean().item():.4f}")

    from torch import nn
    criterion = nn.HuberLoss(delta=0.1)
    loss_dict = KGLatentNetV3.compute_loss(
        out, batch, criterion,
        lambda_delta=0.5, lambda_anchor=0.1,
        lambda_gate_entropy=0.005,
        stage_loss_weight="mild_long_term_weight",
    )
    print(f"    total_loss={loss_dict['total'].item():.4f}")
    print(f"    PASS")

print("\nAll 3 variants smoke test PASSED")
