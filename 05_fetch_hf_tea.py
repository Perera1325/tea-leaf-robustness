"""Download yunusserhat/tea_sickness_dataset (Hugging Face) and save ONLY the classes that overlap
with the Sri Lankan source dataset. Images are not redistributed in this repo.
usage: python 05_fetch_hf_tea.py [out_dir=data/hf_tea]
Mapping: algal_leaf -> Algal_leaf_spot, gray_light -> Gray_blight, healthy -> Healthy
Writes manifests/hf_tea_manifest.csv (path, hf_split, hf_index, hf_class, mapped_class).
Requires: pip install datasets
"""
import csv, pathlib, sys
from datasets import load_dataset

MAP = {"algal_leaf": "Algal_leaf_spot", "gray_light": "Gray_blight", "healthy": "Healthy"}
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/hf_tea")
ds = load_dataset("yunusserhat/tea_sickness_dataset")
names = ds["train"].features["label"].names
rows, counts = [], {}
for split in ds:
    for i, ex in enumerate(ds[split]):
        n = names[ex["label"]]
        counts[n] = counts.get(n, 0) + 1
        if n not in MAP:
            continue
        d = out / MAP[n]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"hf_{split}_{i:04d}.jpg"
        im = ex["image"].convert("RGB")
        im.thumbnail((1024, 1024))   # model uses 224x224 anyway
        im.save(p, quality=95)
        rows.append((p.as_posix(), split, i, n, MAP[n]))
man = pathlib.Path("manifests")
man.mkdir(exist_ok=True)
with open(man / "hf_tea_manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "hf_split", "hf_index", "hf_class", "mapped_class"])
    w.writerows(rows)
print("all classes in HF dataset:", counts)
print("saved", len(rows), "images ->", out)
