from __future__ import annotations
import typing as tp
from pathlib import Path
import yaml

_PRESETS_PATH = Path(__file__).parent / "presets.yaml"
_PRESETS: tp.Optional[tp.Dict[str, tp.Any]] = None

_PATHS_KEYS = {"input", "output_dir", "model", "model_cfg", "device"}


def _load() -> tp.Dict[str, tp.Any]:
    global _PRESETS
    if _PRESETS is None:
        with _PRESETS_PATH.open(encoding="utf-8") as f:
            _PRESETS = yaml.safe_load(f)
    return _PRESETS


def get_analysis_preset(
    str_particleType: str,
    str_magnification: str,
) -> tp.Dict[str, tp.Any]:
    """Return preset dict for particle_type x magnification, or {} if not found."""
    data = _load()
    return dict(data.get(str_particleType, {}).get(str_magnification, {}))


def load_paths_config(str_config_path: str) -> tp.Dict[str, tp.Any]:
    """Load a paths config YAML and return only non-empty recognised keys.

    Unknown keys and empty-string values are silently ignored so they never
    shadow argparse defaults.  Returns {} if the file does not exist.
    """
    try:
        with Path(str_config_path).open(encoding="utf-8") as obj_f:
            obj_raw = yaml.safe_load(obj_f) or {}
    except FileNotFoundError:
        return {}
    return {
        str_k: str_v
        for str_k, str_v in obj_raw.items()
        if str_k in _PATHS_KEYS and isinstance(str_v, str) and str_v.strip()
    }
