"""Stage 2: set aside an independent VERIFICATION set by moving a random sample of training images
(and their labels) into verify/. Run this ONCE, before any training.
  python 00_make_verification_split.py --data dataset/data.yaml --n 100 --seed 0 [--dry-run]
Creates: <dataset>/verify/images, <dataset>/verify/labels, <dataset>/verify.yaml, results/verification_files.txt
Use later:  python 03_verify.py --weights ... --verify-data <dataset>/verify.yaml
Note: images taken of the same pole or scene can look alike. If the dataset has no scene IDs,
mention this as a limitation in your report.
"""
import argparse, random, shutil, yaml
from pathlib import Path
from common import load_data_yaml, split_dir, label_path, list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg, root, names = load_data_yaml(a.data)
    train_dir = split_dir(cfg, root, "train")
    imgs = list_images(train_dir)
    if a.n > 0.2 * len(imgs):
        raise SystemExit(f"--n {a.n} is more than 20% of {len(imgs)} training images. Use a smaller number.")
    vdir = root / "verify"
    if vdir.exists():
        raise SystemExit(f"{vdir} already exists. Verification set was already created; do not repeat.")
    picked = random.Random(a.seed).sample(imgs, a.n)
    print(f"Selected {len(picked)} of {len(imgs)} training images (seed={a.seed}).")
    if a.dry_run:
        print("Dry run only. Nothing moved.")
        return
    (vdir / "images").mkdir(parents=True)
    (vdir / "labels").mkdir()
    for img in picked:
        lp = label_path(img)
        shutil.move(str(img), vdir / "images" / img.name)
        if lp.exists():
            shutil.move(str(lp), vdir / "labels" / lp.name)
    vcfg = {"path": str(root), "train": cfg["train"], "val": "verify/images",
            "names": {i: n for i, n in enumerate(names)}}
    (root / "verify.yaml").write_text(yaml.safe_dump(vcfg, sort_keys=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/verification_files.txt").write_text("\n".join(p.name for p in picked))
    print(f"Moved to {vdir}. Config written to {root / 'verify.yaml'}")


if __name__ == "__main__":
    main()
