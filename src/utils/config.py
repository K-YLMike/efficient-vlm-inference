"""Load ``config.yaml`` and resolve all runtime paths under PROJECT_BASE.

No path is hard-coded. PROJECT_BASE defaults to the current working
directory (which on the cluster is SLURM_SUBMIT_DIR), so the whole project
relocates by setting a single environment variable.
"""

import os
from typing import Any, Dict, List

import yaml


def project_base() -> str:
    """Return PROJECT_BASE, defaulting to the current working directory."""
    return os.environ.get("PROJECT_BASE", os.getcwd())


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config and attach resolved absolute paths.

    The returned dict gains a ``resolved`` block whose values are absolute
    directories rooted at PROJECT_BASE.
    """
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    base = project_base()
    paths = cfg.get("paths", {})
    cfg["resolved"] = {
        "base": base,
        "data_dir": os.path.join(base, paths.get("data_dir", "data")),
        "quantized_dir": os.path.join(
            base, paths.get("quantized_dir", "quantized")),
        "results_dir": os.path.join(
            base, paths.get("results_dir", "results")),
        "calib_dir": os.path.join(
            base, paths.get("calib_dir", "calibration")),
        "docs_dir": os.path.join(base, paths.get("docs_dir", "docs")),
    }
    return cfg


def quant_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of quantization-config dicts from the config."""
    return list(cfg.get("quant_configs", []))


def config_by_name(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a single quant-config dict by its ``name`` field."""
    for entry in quant_configs(cfg):
        if entry["name"] == name:
            return entry
    raise KeyError("unknown quant config: {}".format(name))


def checkpoint_dir(cfg: Dict[str, Any], name: str) -> str:
    """Return the checkpoint directory for a quant config.

    The fp16 baseline has no produced checkpoint; it is served directly
    from the base model, so this returns the base model id for it.
    """
    entry = config_by_name(cfg, name)
    if entry.get("method", "none") == "none":
        return cfg["project"]["base_model"]
    return os.path.join(cfg["resolved"]["quantized_dir"], name)
