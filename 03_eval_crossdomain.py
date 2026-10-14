"""RQ1 + RQ4: accuracy/macro-F1 per split, ECE, temperature scaling, risk-coverage (selective prediction).
usage: python 03_eval_crossdomain.py model.keras
Report: in-domain vs each test_<domain>  -> 'domain-shift drop' table; reliability diagram data; coverage@95%-accuracy.
"""
import sys, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
m = tf.keras.models.load_model(sys.argv[1])
df = pd.read_csv("splits.csv"); classes = open("classes.txt").read().split("\n"); idx = {c: i for i, c in enumerate(classes)}

def logits(split):
    d = df[df.split == split]
    def f(p):
        x = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.image.resize(x, (224, 224))
    t = tf.data.Dataset.from_tensor_slices(d.path.values).map(f).batch(32)
    return m.predict(t, verbose=0), d.cls.map(idx).values

def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = pred == y; e = 0
    for lo, hi in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any(): e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return e

def coverage_at(p, y, target=0.95):
    conf = p.max(1); order = np.argsort(-conf); correct = (p.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y)+1)
    ok = np.where(cum >= target)[0]
    return (ok.max()+1) / len(y) if len(ok) else 0.0

zv, yv = logits("val")
T = minimize_scalar(lambda t: -log_softmax(zv / t, axis=1)[np.arange(len(yv)), yv].mean(), bounds=(0.05, 10), method="bounded").x
print(f"temperature T={T:.3f} (fitted on VAL only)")
rows = []
for s in sorted(df.split.unique()):
    if s in ("train", "val"): continue
    z, y = logits(s); p0, p1 = softmax(z, 1), softmax(z / T, 1)
    rows.append(dict(split=s, n=len(y), acc=(p0.argmax(1) == y).mean(), macroF1=f1_score(y, p0.argmax(1), average="macro"),
                     ECE_raw=ece(p0, y), ECE_temp=ece(p1, y), cov_at_95acc=coverage_at(p1, y)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv(sys.argv[1] + ".results.csv", index=False)
