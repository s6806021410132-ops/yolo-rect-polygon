"""Build a 5-class subset of COCO 2017 in YOLO format, twice:
  datasets/coco5-det  -> Rectangle labels (class cx cy w h)
  datasets/coco5-seg  -> Polygon labels   (class x1 y1 x2 y2 ...)
Both use exactly the same images and objects so the two models are comparable.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from ultralytics.data.converter import merge_multi_segment

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "datasets"

# COCO category id -> our class index
CLASSES = {1: "person", 3: "car", 4: "motorcycle", 17: "cat", 18: "dog"}
CAT2IDX = {cid: i for i, cid in enumerate(CLASSES)}
PER_CLASS = {"train2017": 260, "val2017": 60}  # images sampled per class
SEED = 0


def load(split):
    data = json.loads((RAW / "annotations" / f"instances_{split}.json").read_text())
    imgs = {im["id"]: im for im in data["images"]}
    anns = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] in CLASSES and not a["iscrowd"]:
            anns[a["image_id"]].append(a)
    return imgs, anns


def select(imgs, anns, n_per_class):
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for iid, al in anns.items():
        for cid in {a["category_id"] for a in al}:
            by_class[cid].append(iid)
    chosen = set()
    # rarest classes first so person/car don't eat the budget
    for cid in sorted(by_class, key=lambda c: len(by_class[c])):
        pool = sorted(by_class[cid])
        rng.shuffle(pool)
        have = sum(1 for i in chosen if any(a["category_id"] == cid for a in anns[i]))
        for iid in pool:
            if have >= n_per_class:
                break
            if iid not in chosen:
                chosen.add(iid)
                have += 1
    return sorted(chosen)


def det_line(a, w, h):
    x, y, bw, bh = a["bbox"]
    return f"{CAT2IDX[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def seg_line(a, w, h):
    segs = a["segmentation"]
    if len(segs) > 1:
        s = np.concatenate(merge_multi_segment(segs), axis=0)
    else:
        s = np.array(segs[0]).reshape(-1, 2)
    s = (s / np.array([w, h])).clip(0, 1).reshape(-1)
    return f"{CAT2IDX[a['category_id']]} " + " ".join(f"{v:.6f}" for v in s)


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return
        except Exception as e:  # retry flaky downloads
            err = e
    print("FAILED", url, err, file=sys.stderr)


def main():
    stats = {}
    for split, n in PER_CLASS.items():
        name = "train" if split.startswith("train") else "val"
        imgs, anns = load(split)
        ids = select(imgs, anns, n)
        img_dir = OUT / "images_cache" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(16) as ex:
            for iid in ids:
                im = imgs[iid]
                ex.submit(download, im["coco_url"], img_dir / im["file_name"])

        counts = Counter()
        for kind, fn in (("det", det_line), ("seg", seg_line)):
            (OUT / f"coco5-{kind}" / "images" / name).mkdir(parents=True, exist_ok=True)
            (OUT / f"coco5-{kind}" / "labels" / name).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            im = imgs[iid]
            src = img_dir / im["file_name"]
            if not src.exists():
                continue
            w, h = im["width"], im["height"]
            for kind, fn in (("det", det_line), ("seg", seg_line)):
                dst = OUT / f"coco5-{kind}" / "images" / name / im["file_name"]
                if not dst.exists():
                    os.link(src, dst)
                lines = [fn(a, w, h) for a in anns[iid]]
                (OUT / f"coco5-{kind}" / "labels" / name / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
            counts.update(CLASSES[a["category_id"]] for a in anns[iid])
        stats[name] = {"images": len(ids), "objects_per_class": dict(counts)}
        print(name, stats[name])

    for kind in ("det", "seg"):
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES.values()))
        (OUT / f"coco5-{kind}.yaml").write_text(
            f"path: {(OUT / f'coco5-{kind}').as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
        )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
