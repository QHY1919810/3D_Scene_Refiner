# -*- coding: utf-8 -*-
"""
register_all.py (robust)

Goals:
- Recursively import all python files under this extension so @threestudio.register(...) executes.
- Handle "duplicate registry name" conflicts WITHOUT aborting import:
    - During import, temporarily monkeypatch threestudio.register to a "safe" version that
      catches ValueError("... already exists") and simply returns the class/function.
      This lets modules finish executing so symbols like GaussianBaseModel exist, while keeping
      the first-registered implementation in the global registry.
- Skip obvious backup/tmp files to avoid accidental duplicates.

This file is designed to work with the custom loader in launch.py that imports only:
  custom/<ext>/__init__.py
So we run register_all() at import time.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, List


_THIS_DIR = Path(__file__).resolve().parent


def _is_backup_file(p: Path) -> bool:
    s = p.name.lower()
    stem = p.stem.lower()
    bad_tokens = ["bak", "backup", "old", "copy", "tmp", "temp"]
    if any(t in s for t in bad_tokens) or any(t in stem for t in bad_tokens):
        return True
    if p.name.startswith("_"):
        return True
    return False


def _iter_py_files(root: Path, subdirs: List[str]) -> Iterable[Path]:
    for sd in subdirs:
        d = root / sd
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            if _is_backup_file(p):
                continue
            yield p


def _safe_register_wrapper():
    """
    Return (safe_register, restore_fn).
    safe_register has the same signature as threestudio.register.
    """
    import threestudio  # local import
    orig_register = threestudio.register

    def safe_register(name: str):
        deco = orig_register(name)

        def wrapped(obj):
            try:
                return deco(obj)
            except ValueError as e:
                msg = str(e)
                # registry conflict -> keep existing, but allow module to load
                if "already exists" in msg and "Names of extensions conflict" in msg:
                    print(f"[ts3dgs.register_all] WARN: duplicate registry name '{name}', keeping existing. ({msg})")
                    return obj
                raise

        return wrapped

    def restore():
        threestudio.register = orig_register

    return safe_register, restore


def import_module_from_file(fullname: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(fullname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    # IMPORTANT: dataclass + relative imports stability
    sys.modules[fullname] = module

    # Temporarily make threestudio.register safe so duplicates don't kill imports.
    import threestudio
    safe_register, restore = _safe_register_wrapper()
    threestudio.register = safe_register
    try:
        spec.loader.exec_module(module)
    finally:
        restore()
    return module


def register_all():
    # Synthetic namespace so relative imports can resolve consistently during file-path import.
    if "ts3dgs" not in sys.modules:
        pkg = ModuleType("ts3dgs")
        pkg.__path__ = [str(_THIS_DIR)]
        sys.modules["ts3dgs"] = pkg

    # Import order: foundations first.
    subdirs = [
        "utils",
        "prompt_processors",
        "data",
        "geometry",
        "renderer",
        "material",
        "background",
        "system",
        "systems",
        "guidance",
    ]

    for f in _iter_py_files(_THIS_DIR, subdirs):
        rel = f.relative_to(_THIS_DIR).with_suffix("")
        modname = "ts3dgs." + ".".join(rel.parts)
        try:
            import_module_from_file(modname, f)
        except Exception as e:
            # For debugging: show which file failed
            raise RuntimeError(f"[ts3dgs.register_all] Failed importing {f}: {e}") from e


# Run immediately when imported by custom loader
register_all()
