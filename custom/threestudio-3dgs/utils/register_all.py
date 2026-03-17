"""
Bootstrap loader for custom/threestudio-3dgs (hyphenated folder).

Why this exists:
- The folder name contains a '-' so it cannot be imported as a normal Python package.
- threestudio's custom module loader typically imports only top-level .py files under each custom module.
- Our important registrations (DataModule, prompt processors, utils) live in subfolders (data/, prompt_processors/, utils/),
  so they may never get imported -> threestudio.find(...) KeyError.

What this file does:
- Recursively imports selected submodules by **file path** into a stable namespace (ts3dgs.*).
- Ensures sys.modules[name] is set before exec_module (avoids dataclass-related crashes).
- Running this once is enough to register:
  - colmap-scene-datamodule
  - plain-prompt-processor
  - any other @threestudio.register(...) classes in those subfolders
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Iterable, List


_LOADED_FLAG = "TS3DGS_REGISTER_ALL_LOADED"


def import_module_from_file(module_name: str, file_path: Path):
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Critical: register in sys.modules BEFORE executing (dataclass safety)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_namespace(ns: str):
    if ns in sys.modules:
        return
    parts = ns.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _iter_py_files(base: Path, rel_dirs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for d in rel_dirs:
        p = base / d
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.py")):
            if f.name.startswith("_"):
                continue
            if f.name == "__init__.py":
                continue
            if "__pycache__" in str(f):
                continue
            files.append(f)
    return files


def register_all():
    # guard
    if getattr(sys.modules.get(__name__), _LOADED_FLAG, False):
        return
    setattr(sys.modules.get(__name__), _LOADED_FLAG, True)

    base = Path(__file__).resolve().parent

    # stable namespace for these file-path imports
    _ensure_namespace("ts3dgs")

    # Only import the subfolders we rely on for observed supervision
    targets = _iter_py_files(base, ["data", "prompt_processors", "utils"])

    for f in targets:
        # Map file path -> module name under ts3dgs.<subdir>.<stem>
        # e.g. custom/threestudio-3dgs/data/colmap_scene_datamodule.py -> ts3dgs.data.colmap_scene_datamodule
        rel = f.relative_to(base)
        parts = rel.parts[:-1]  # dirs
        stem = f.stem
        modname = ".".join(["ts3dgs", *parts, stem])
        if modname in sys.modules:
            continue
        try:
            import_module_from_file(modname, f)
        except Exception as e:
            raise RuntimeError(f"[ts3dgs.register_all] Failed importing {f}: {e}") from e


# Execute on import so threestudio custom loader just needs to import this file.
register_all()
