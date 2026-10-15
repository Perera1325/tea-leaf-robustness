import sys, time, os, numpy as np, pandas as pd, tensorflow as tf
stem = sys.argv[1]
df = pd.read_csv("splits.csv"); classes = open("classes.txt").read().split("\n"); idx = {c: i for i, c in enumerate(classes)}
def load(p):
    x = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
    return tf.image.resize(x, (224, 224)).numpy().astype(np.float32)[None]
splits = ["test_indomain", "test_hf_tea", "test_mendeley_ext"]
X, Y = {}, {}
for s in splits:
    d = df[df.split == s]; X[s] = [load(p) for p in d.path]; Y[s] = d.cls.map(idx).values
rows, ref = [], {}
for v in ["fp32", "fp16", "int8"]:
    f = f"{stem}_{v}.tflite"
    it = tf.lite.Interpreter(model_path=f, num_threads=1, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES); it.allocate_tensors()
    i_in = it.get_input_details()[0]["index"]; i_out = it.get_output_details()[0]["index"]
    def run(x):
        it.set_tensor(i_in, x); it.invoke(); return it.get_tensor(i_out)[0]
    for _ in range(5): run(X[splits[0]][0])
    t = []
    for k in range(50):
        t0 = time.perf_counter(); run(X[splits[0]][k % len(X[splits[0]])]); t.append((time.perf_counter() - t0) * 1000)
    for s in splits:
        pred = np.array([run(x).argmax() for x in X[s]])
        if v == "fp32": ref[s] = pred
        rows.append(dict(variant=v, split=s, n=len(pred), acc=round((pred == Y[s]).mean(), 3),
                         agree_with_fp32=round((pred == ref[s]).mean(), 3),
                         size_MB=round(os.path.getsize(f) / 1e6, 2), latency_ms_median=round(float(np.median(t)), 1)))
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(f"{stem}.tflite_results.csv", index=False)
