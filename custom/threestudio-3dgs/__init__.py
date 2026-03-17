import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )


from .background import gaussian_mvdream_background
from .geometry import exporter, gaussian_base, gaussian_io
from .material import gaussian_material
from .renderer import (
    diff_gaussian_rasterizer,
    diff_gaussian_rasterizer_advanced,
    diff_gaussian_rasterizer_background,
    diff_gaussian_rasterizer_shading,
)
from .system import gaussian_mvdream, gaussian_splatting, gaussian_zero123

from .register_all import register_all 


import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_reg = _here / "register_all.py"

spec = importlib.util.spec_from_file_location("ts3dgs_register_all", str(_reg))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load register_all from {_reg}")
mod = importlib.util.module_from_spec(spec)
# Important: insert into sys.modules BEFORE exec_module (dataclasses may rely on it)
sys.modules["ts3dgs_register_all"] = mod
spec.loader.exec_module(mod)  # type: ignore[attr-defined]


