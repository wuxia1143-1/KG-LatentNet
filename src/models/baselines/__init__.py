from __future__ import annotations

from src.models.baselines.dhgas_adapter import DHGASAdapter
from src.models.baselines.graphcare_adapter import GraphCareAdapter
from src.models.baselines.hyperimts_adapter import HyperIMTSAdapter
from src.models.baselines.kedgn_adapter import KEDGNAdapter
from src.models.baselines.tgnn4i_adapter import TGNN4IAdapter
from src.models.baselines.trans import TRANSModel


BASELINE_MODEL_REGISTRY = {
    "hyperimts": HyperIMTSAdapter,
    "trans": TRANSModel,
    "tgnn4i": TGNN4IAdapter,
    "dhgas": DHGASAdapter,
    "kedgn": KEDGNAdapter,
    "graphcare": GraphCareAdapter,
}
