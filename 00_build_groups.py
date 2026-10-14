"""Group near-duplicate photos (same leaf / same shot) so they never straddle train/test.
usage: python 00_build_groups.py <standardised_raw> <out_data_dir> [domain=mendeley] [hamming=10]
Reads  <standardised_raw>/<Class>/*.jpg
Writes <out_data_dir>/<domain>/<Class>/g<ID>__<orig>.jpg  (copies)  + groups_report.csv
Method: 16x16 difference hash (256 bits) per image; images within `hamming` bits are linked
(union-find); each connected component = one group. Tune `hamming` by LOOKING at merged groups.
Limits: catches near-duplicates / same-shot variants, NOT different photos of the same leaf from other angles.
"""
import sys, shutil, pathlib, csv
import numpy as np
from PIL import Image

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
domain = sys.argv[3] if len(sys.argv) > 3 else "mendeley"
T = int(sys.argv[4]) if len(sys.argv) > 4 else 10

def dhash(p, n=16):
    im = Image.open(p).convert("L").resize((n + 1, n), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

files = sorted(f for c in src.iterdir() if c.is_dir() for f in c.iterdir()
               if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
H = np.array([dhash(f) for f in files])
par = list(range(len(files)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
for i in range(len(files)):                       # O(n^2) is fine for ~2k images
    d = (H[i + 1:] != H[i]).sum(1)
    for j in np.nonzero(d <= T)[0]:
        if files[i].parent == files[i + 1 + j].parent:   # only link within the same class
            par[find(i)] = find(i + 1 + j)
roots = {}
rows = []
for i, f in enumerate(files):
    g = roots.setdefault(find(i), len(roots))
    dst = out / domain / f.parent.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst / f"g{g:04d}__{f.name}")
    rows.append((f.parent.name, f.name, g))
with open("groups_report.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("class", "file", "group")] + rows)
sizes = np.bincount([r[2] for r in rows])
print(f"{len(files)} images -> {len(sizes)} groups; largest group = {sizes.max()}; "
      f"singletons = {(sizes == 1).sum()}; groups with >1 image = {(sizes > 1).sum()}")
