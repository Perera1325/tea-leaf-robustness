"""Near-duplicate check: external images vs reference sets (dHash, 256 bits)."""
import sys, pathlib, numpy as np
from PIL import Image

def dhash(p):
    im = Image.open(p)
    im.draft("L", (256, 256))
    a = np.asarray(im.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def load(d):
    fs = sorted(q for q in pathlib.Path(d).rglob("*")
                if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return fs, np.stack([dhash(q) for q in fs])

_, E = load(sys.argv[1])
for ref in sys.argv[2:]:
    rf, R = load(ref)
    best = np.full(len(E), 999)
    for j in range(0, len(R), 256):
        d = (E[:, None, :] != R[None, j:j + 256, :]).sum(-1)
        best = np.minimum(best, d.min(1))
    print(f"{ref}: {len(E)} ext vs {len(R)} ref | min Hamming = {best.min()} bits"
          f" | <=10 bits: {(best <= 10).sum()} | <=25 bits: {(best <= 25).sum()}")
