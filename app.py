
import argparse
import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

model = None

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/detect")
def detect():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image received. Choose a JPG or PNG file."), 400

    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="This file could not be read as an image."), 400

    conf = float(request.form.get("conf", 0.25))
    r = model.predict(img, conf=conf, verbose=False)[0]

    dets = [
        {
            "class": r.names[int(b.cls)],
            "confidence": round(float(b.conf), 3),
            "box": [round(float(v)) for v in b.xyxy[0]]
        }
        for b in r.boxes
    ]

    ok, png = cv2.imencode(".jpg", r.plot())

    return jsonify(
        image="data:image/jpeg;base64," + base64.b64encode(png).decode(),
        detections=dets,
        ms=round(r.speed["inference"], 1)
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs/exp1_baseline/weights/best.pt")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    model = YOLO(a.weights)
    app.run(host="0.0.0.0", port=a.port)
