"""After training: validate, predict on val samples, export ONNX for the web app, collect results."""
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WEB = ROOT / "docs"
PLOTS = ["results.png", "confusion_matrix_normalized.png", "labels.jpg", "val_batch0_pred.jpg", "val_batch0_labels.jpg",
         "BoxPR_curve.png", "MaskPR_curve.png", "results.csv", "args.yaml"]


def main():
    val_imgs = sorted((ROOT / "datasets" / "coco5-seg" / "images" / "val").glob("*.jpg"))
    random.Random(1).shuffle(val_imgs)
    samples = val_imgs[:12]

    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "samples").mkdir(parents=True, exist_ok=True)
    (ROOT / "weights").mkdir(exist_ok=True)
    for i, p in enumerate(samples[:6], 1):
        shutil.copy(p, WEB / "samples" / f"{i}.jpg")

    summary = {}
    for kind in ("det", "seg"):
        run = ROOT / "runs" / kind
        best = run / "weights" / "best.pt"
        shutil.copy(best, ROOT / "weights" / f"yolo11n-{kind}-coco5.pt")
        out = RES / kind
        out.mkdir(parents=True, exist_ok=True)
        for f in PLOTS:
            if (run / f).exists():
                shutil.copy(run / f, out / f)

        model = YOLO(best)
        m = model.val(data=str(ROOT / "datasets" / f"coco5-{kind}.yaml"), project=str(ROOT / "runs"), name=f"{kind}-val", exist_ok=True)
        summary[kind] = {"box_mAP50": m.box.map50, "box_mAP50-95": m.box.map,
                         "per_class_box_mAP50-95": dict(zip(model.names.values(), map(float, m.box.maps)))}
        if kind == "seg":
            summary[kind] |= {"mask_mAP50": m.seg.map50, "mask_mAP50-95": m.seg.map}

        model.predict(samples, save=True, project=str(RES / kind), name="predict", exist_ok=True, conf=0.25)

        onnx = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
        shutil.copy(onnx, WEB / "models" / f"{kind}.onnx")
        shutil.copy(onnx, ROOT / "weights" / f"yolo11n-{kind}-coco5.onnx")

    (RES / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()

