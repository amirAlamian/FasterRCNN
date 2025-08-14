from pathlib import Path
import cv2, shutil
from typing import List
from .config import RESULTS_DIR, CLUSTERS_DIR, PLOTS_DIR


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def list_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])

def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def copy_to(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
