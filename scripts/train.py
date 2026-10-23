"""Train both models: python scripts/train.py det|seg"""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
kind = sys.argv[1]
base = {"det": "yolo11n.pt", "seg": "yolo11n-seg.pt"}[kind]

if __name__ == "__main__":
    model = YOLO(base)
    model.train(
        data=str(ROOT / "datasets" / f"coco5-{kind}.yaml"),
        epochs=60,
        imgsz=640,
        batch=16,
        workers=4,
        project=str(ROOT / "runs"),
        name=f"{kind}",
        exist_ok=True,
        seed=0,
        plots=True,
    )
