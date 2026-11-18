"""Stage 5: verify the chosen model on images NOT used for training, validation or model selection.
  python 03_verify.py --weights runs/exp1_baseline/weights/best.pt --images verify_images/
Optional, if the verification images have YOLO labels (gives real metrics):
  python 03_verify.py --weights ... --verify-data verify.yaml
"""
import argparse, csv
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images")
    ap.add_argument("--verify-data")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    model = YOLO(a.weights)
    if a.verify_data:
        m = model.val(data=a.verify_data, project="runs", name="verification", exist_ok=True)
        print(f"P={m.box.mp:.4f} R={m.box.mr:.4f} mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")
    if a.images:
        Path("results").mkdir(exist_ok=True)
        with open("results/verification_detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2", "reviewer_verdict"])
            for r in model.predict(a.images, conf=a.conf, save=True, project="runs",
                                   name="verification_images", exist_ok=True, stream=True):
                for b in r.boxes:
                    x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0]]
                    w.writerow([Path(r.path).name, r.names[int(b.cls)], round(float(b.conf), 3),
                                x1, y1, x2, y2, ""])
        print("Annotated images: runs/verification_images. Fill 'reviewer_verdict' "
              "(correct / false alarm) in results/verification_detections.csv; note missed objects separately.")


if __name__ == "__main__":
    main()
