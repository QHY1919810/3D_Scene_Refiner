"""ts3dgs custom module bootstrap (file-path importer)

Goal: ensure all @threestudio.register(...) in custom/threestudio-3dgs are executed,
so threestudio.find(...) works for keys like:
  - colmap-scene-datamodule
  - plain-prompt-processor
  - gaussian-splatting-system
  - gaussian-splatting (geometry)
  - diff-gaussian-rasterizer (renderer)

Why needed:
- Folder name contains '-' so normal Python imports are awkward.
- launch.py imports only the top-level custom module; subfolders may never load.

Strategy:
- Create a synthetic import namespace: ts3dgs.*
- Import selected subfolders by file-path under that namespace.
- Ensure sys.modules is populated *before* exec_module (dataclass safety).
- Ensure parent packages (ts3dgs.data, ts3dgs.systems, ...) exist with __path__ so
  intra-module relative imports (from ..utils ...) work.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Iterable, List

_LOADED_FLAG = "TS3DGS_REGISTER_ALL_LOADED"


def _mk_pkg(name: str, path: Path):
    """Create a synthetic package module with __path__ so relative imports work."""
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(path)]  # type: ignore[attr-defined]
    m.__package__ = name
    sys.modules[name] = m


def import_module_from_file(module_name: str, file_path: Path):
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Critical: put in sys.modules BEFORE exec (dataclass safety)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _iter_py_files(base: Path, rel_dirs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for d in rel_dirs:
        p = base / d
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.py")):
            if f.name == "__init__.py":
                continue
            if f.name.startswith("_"):
                continue
            if "__pycache__" in str(f):
                continue
            files.append(f)
    return files


def register_all():
    mod = sys.modules.get(__name__)
    if mod is not None and getattr(mod, _LOADED_FLAG, False):
        return
    if mod is not None:
        setattr(mod, _LOADED_FLAG, True)

    base = Path(__file__).resolve().parent

    # root synthetic namespace
    _mk_pkg("ts3dgs", base)

    # Import order: utils first helps other modules
    subdirs = [
        "utils",
        "prompt_processors",
        "data",
        "geometry",
        "renderer",
        "systems",
    ]

    targets = _iter_py_files(base, subdirs)

    for f in targets:
        rel = f.relative_to(base)
        parts = list(rel.parts[:-1])  # directories
        stem = f.stem

        # Ensure parent packages exist with correct __path__
        cur_path = base
        pkg_name = "ts3dgs"
        for d in parts:
            cur_path = cur_path / d
            pkg_name = pkg_name + "." + d
            _mk_pkg(pkg_name, cur_path)

        modname = ".".join(["ts3dgs", *parts, stem])
        if modname in sys.modules:
            continue
        import_module_from_file(modname, f)


# Execute on import
register_all()
