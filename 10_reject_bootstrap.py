import sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax

def fit_temperature(z, y):
    return minimize_scalar(lambda t: -log_softmax(z / t, axis=1)[np.arange(len(y)), y].mean(),
                           bounds=(0.05, 10), method="bounded").x

def pick_threshold(p, y, target=0.95):
    conf = p.max(1); order = np.argsort(-conf)
    correct = (p.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    ok = np.where(cum >= target)[0]
    return float(conf[order][ok.max()]) if len(ok) else 1.01

def metrics(p, y, tau):
    correct = p.argmax(1) == y; acc = p.max(1) >= tau; wrong = ~correct
    return dict(acc=correct.mean(), coverage=acc.mean(),
                sel_acc=correct[acc].mean() if acc.any() else np.nan,
                error_catch=(wrong & ~acc).sum() / wrong.sum() if wrong.any() else np.nan)

def bootstrap(p, y, tau, B=1000, seed=0):
    rng = np.random.default_rng(seed); out = {k: [] for k in ("acc","coverage","sel_acc","error_catch")}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); m = metrics(p[i], y[i], tau)
        for k in out: out[k].append(m[k])
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in out.items()}

def get_logits(model, df, idx, split):
    import tensorflow as tf
    d = df[df.split == split]
    def f(path):
        x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(x, (224, 224))
    t = tf.data.Dataset.from_tensor_slices(d.path.values).map(f).batch(32)
    return model.predict(t, verbose=0), d.cls.map(idx).values

if __name__ == "__main__":
    import tensorflow as tf
    path = sys.argv[1]; target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    model = tf.keras.models.load_model(path)
    df = pd.read_csv("splits.csv"); classes = open("classes.txt").read().split("\n"); idx = {c: i for i, c in enumerate(classes)}
    zv, yv = get_logits(model, df, idx, "val")
    T = fit_temperature(zv, yv); pv = softmax(zv / T, 1); tau = pick_threshold(pv, yv, target)
    mv = metrics(pv, yv, tau)
    print(f"T={T:.3f} | tau={tau:.3f} chosen on VAL (val coverage {mv['coverage']:.3f}, val selective acc {mv['sel_acc']:.3f})")
    rows = []
    for s in sorted(df.split.unique()):
        if s in ("train", "val"): continue
        z, y = get_logits(model, df, idx, s); p = softmax(z / T, 1)
        m = metrics(p, y, tau); ci = bootstrap(p, y, tau)
        g = lambda k: f"{m[k]:.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]"
        rows.append(dict(split=s, n=len(y), acc=g("acc"), coverage=g("coverage"),
                         selective_acc=g("sel_acc"), errors_caught=g("error_catch")))
    out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(path + ".reject.csv", index=False)
