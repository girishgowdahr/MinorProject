"""Shared helpers. Class names are always read from the dataset's data.yaml (never hard-coded)."""
from pathlib import Path
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return cfg, root, names


def split_dir(cfg, root, split):
    """Return the images folder for a split, or None if the split is not defined."""
    value = cfg.get(split)
    if not value:
        return None
    p = Path(value if isinstance(value, str) else value[0])
    return p if p.is_absolute() else (root / p)


def label_path(image_path):
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt"""
    parts = list(image_path.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
