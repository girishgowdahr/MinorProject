"""Stage 3-4: train and evaluate a small set of YOLO configurations on the SAME dataset and splits.
Edit EXPERIMENTS after dataset inspection (change ONE factor at a time). Nothing here is a result;
numbers appear only after you run it.
Usage: python 02_run_experiments.py --data path/to/data.yaml [--epochs 50] [--device cpu|0]
"""
import argparse, csv
from pathlib import Path
from ultralytics import YOLO
from common import load_data_yaml

# Example design: baseline vs larger model vs higher resolution. Adjust to your compute.
EXPERIMENTS = [
    {"name": "exp1_baseline", "model": "yolov8n.pt", "imgsz": 640},
    {"name": "exp2_larger_model", "model": "yolov8s.pt", "imgsz": 640},
    {"name": "exp3_higher_res", "model": "yolov8n.pt", "imgsz": 960},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="run just one experiment by name")
    a = ap.parse_args()
    cfg, root, _ = load_data_yaml(a.data)
    eval_split = "test" if cfg.get("test") else "val"
    print(f"Final metrics will be computed on the '{eval_split}' split.")
    out = Path("results"); out.mkdir(exist_ok=True)
    csv_path = out / "experiment_results.csv"
    fields = ["experiment", "model", "imgsz", "epochs", "eval_split", "precision", "recall",
              "mAP50", "mAP50_95", "F1", "inference_ms_per_image", "weights"]
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for e in EXPERIMENTS:
            if a.only and e["name"] != a.only:
                continue
            model = YOLO(e["model"])
            model.train(data=a.data, epochs=a.epochs, imgsz=e["imgsz"], batch=a.batch,
                        seed=a.seed, device=a.device, project="runs", name=e["name"], exist_ok=True)
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            m = YOLO(str(best)).val(data=a.data, split=eval_split, imgsz=e["imgsz"], device=a.device,
                                    project="runs", name=e["name"] + "_eval", exist_ok=True)
            p, r = float(m.box.mp), float(m.box.mr)
            w.writerow({"experiment": e["name"], "model": e["model"], "imgsz": e["imgsz"],
                        "epochs": a.epochs, "eval_split": eval_split, "precision": round(p, 4),
                        "recall": round(r, 4), "mAP50": round(float(m.box.map50), 4),
                        "mAP50_95": round(float(m.box.map), 4),
                        "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                        "inference_ms_per_image": round(m.speed["inference"], 2), "weights": str(best)})
            f.flush()
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
