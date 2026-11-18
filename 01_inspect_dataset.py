"""Stage 1-2: inspect a YOLO-format dataset BEFORE finalizing detection classes.
Usage: python 01_inspect_dataset.py --data path/to/data.yaml
Reports: image/label counts, missing or empty labels, invalid label lines,
instances per class, and images shared between splits (data leakage).
"""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from common import load_data_yaml, split_dir, label_path, list_images


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/inspection_report.json")
    a = ap.parse_args()
    cfg, root, names = load_data_yaml(a.data)
    report, hashes = {"classes_in_yaml": names, "splits": {}}, defaultdict(set)
    for split in ("train", "val", "test"):
        folder = split_dir(cfg, root, split)
        if folder is None or not folder.exists():
            report["splits"][split] = "not defined or folder missing"
            continue
        imgs = list_images(folder)
        counts, missing, empty, bad = Counter(), 0, 0, 0
        for img in imgs:
            hashes[md5(img)].add(split)
            lp = label_path(img)
            if not lp.exists():
                missing += 1
                continue
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for parts in lines:
                try:
                    cls, coords = int(parts[0]), list(map(float, parts[1:5]))
                    ok = 0 <= cls < len(names) and len(coords) == 4 and all(0 <= c <= 1 for c in coords)
                except (ValueError, IndexError):
                    ok = False
                if ok:
                    counts[names[cls]] += 1
                else:
                    bad += 1
        report["splits"][split] = {
            "images": len(imgs), "missing_label_files": missing, "empty_label_files": empty,
            "invalid_label_lines": bad, "instances_per_class": dict(counts)}
    leaks = sum(1 for s in hashes.values() if len(s) > 1)
    report["identical_images_across_splits"] = leaks
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {a.out}. Review this report, then finalize the detection classes.")


if __name__ == "__main__":
    main()
