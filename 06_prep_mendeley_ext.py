"""Prepare Mendeley rrff4vb35m (Real-Life Dataset of Tea Plant Leaf Disease
Classification) as a second external TEST domain. Never used for training.
Resizes to max 1024 px, maps only clearly matching classes, writes a manifest."""
import sys, csv, pathlib
from PIL import Image, ImageOps

SRC = pathlib.Path(sys.argv[1])          # the 'Tea Original Dataset' folder
OUT = pathlib.Path("data/mendeley_ext")
MAP = {"Tea Algal Spot": "Algal_leaf_spot",
       "Tea Brown Blight": "Brown_blight",
       "Tea Grey Blight": "Gray_blight",
       "Tea Healthy": "Healthy"}         # 'Tea Red Spot' skipped on purpose
known = {r["cls"] for r in csv.DictReader(open("splits.csv"))}
rows = []
for folder, cls in MAP.items():
    if cls not in known:
        print("SKIP (class not in splits.csv):", cls)
        continue
    files = sorted(p for p in (SRC / folder).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    (OUT / cls).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((1024, 1024))
        out = OUT / cls / f"mx_{cls}_{i:04d}.jpg"
        im.save(out, quality=90)
        rows.append((out.as_posix(), p.name, folder, cls))
    print(cls, len(files))
pathlib.Path("manifests").mkdir(exist_ok=True)
with open("manifests/mendeley_ext_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "orig_name", "orig_folder", "mapped_class"])
    w.writerows(rows)
print("total", len(rows))
