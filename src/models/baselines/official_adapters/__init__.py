from __future__ import annotations

from src.models.baselines.official_adapters.dhgas_official_adapter import DHGASOfficialAdapter
from src.models.baselines.official_adapters.graphcare_official_adapter import GraphCareOfficialAdapter
from src.models.baselines.official_adapters.hyperimts_official_adapter import HyperIMTSOfficialAdapter
from src.models.baselines.official_adapters.kedgn_official_adapter import KEDGNOfficialAdapter
from src.models.baselines.official_adapters.tgnn4i_official_adapter import TGNN4IOfficialAdapter
from src.models.baselines.official_adapters.trans_official_adapter import TRANSOfficialAdapter


OFFICIAL_BASELINE_REGISTRY = {
    "hyperimts": HyperIMTSOfficialAdapter,
    "trans": TRANSOfficialAdapter,
    "tgnn4i": TGNN4IOfficialAdapter,
    "dhgas": DHGASOfficialAdapter,
    "kedgn": KEDGNOfficialAdapter,
    "graphcare": GraphCareOfficialAdapter,
}
