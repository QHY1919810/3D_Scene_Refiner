from __future__ import annotations

"""COLMAP TXT IO helpers.

These are thin parsers for COLMAP's text model format:
- cameras.txt: 1 line per camera
- images.txt: 2 lines per image (second line is 2D points or blank)

We only parse the fields we need for rendering/training (intrinsics + pose).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass(frozen=True)
class ColmapImage:
    id: int
    qvec_wxyz: List[float]  # [w, x, y, z]
    tvec: List[float]       # [tx, ty, tz]
    camera_id: int
    name: str


def read_cameras_txt(path: str) -> Dict[int, ColmapCamera]:
    cams: Dict[int, ColmapCamera] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cams[cid] = ColmapCamera(
            id=cid,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(x) for x in parts[4:]],
        )
    return cams


def read_images_txt(path: str) -> Dict[int, ColmapImage]:
    ims: Dict[int, ColmapImage] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        iid = int(parts[0])
        q = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        t = [float(parts[5]), float(parts[6]), float(parts[7])]
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        ims[iid] = ColmapImage(
            id=iid,
            qvec_wxyz=q,
            tvec=t,
            camera_id=cam_id,
            name=name,
        )

        # COLMAP format: a second line with POINTS2D[] follows each image line.
        # Many exported files leave it blank; if so, skip it.
        if i < len(lines) and lines[i].strip() == "":
            i += 1

    return ims
