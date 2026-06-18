from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/root/KG_LatentNet_Project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_helper():
    path = ROOT / "scripts" / "honest_real_final_outputs_validation_top.py"
    spec = importlib.util.spec_from_file_location("helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def main() -> None:
    names = [str(x) for x in HELPER.load_tabular(0, "train")["feature_names"]]
    tokens = [
        "chemo",
        "radio",
        "immun",
        "target",
        "therapy",
        "treatment",
        "化疗",
        "放疗",
        "免疫",
        "靶向",
        "治疗",
    ]
    hits = [name for name in names if any(token.lower() in name.lower() for token in tokens)]
    print(json.dumps({"n_features": len(names), "n_hits": len(hits), "hits": hits[:300]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
