"""One-off, idempotent: adds the 'padaug' strategy and the '--extra' multi-source option to 02_train.py."""
s = open("02_train.py", encoding="utf-8").read()
PAD = '''        if a.strategy == "padaug":     # leaf shrunk onto a plain light background
            def _pad(x):
                sc = tf.random.uniform([], 0.45, 1.0)
                n = tf.cast(sc * IMG, tf.int32)
                small = tf.image.resize(x, (n, n))
                col = tf.random.uniform([3], 150.0, 255.0)
                oy = tf.random.uniform([], 0, IMG - n + 1, tf.int32)
                ox = tf.random.uniform([], 0, IMG - n + 1, tf.int32)
                pads = [[oy, IMG - n - oy], [ox, IMG - n - ox], [0, 0]]
                leaf = tf.pad(small, pads)
                mask = tf.pad(tf.ones((n, n, 1)), pads)
                return tf.ensure_shape(leaf + col * (1.0 - mask), [IMG, IMG, 3])
            x = tf.cond(tf.random.uniform([]) < 0.5, lambda: _pad(x), lambda: x)
'''
FA = '        if a.strategy == "fieldaug":'
if 'a.strategy == "padaug"' not in s:
    assert s.count(FA) == 1, "fieldaug anchor not found exactly once"
    s = s.replace(FA, PAD + FA)
if '--extra' not in s:
    T = 'ap.add_argument("--tag", default="")\n'
    assert s.count(T) == 1, "--tag anchor not found"
    s = s.replace(T, T + 'ap.add_argument("--extra", default="")   # e.g. hf_tea or mendeley_ext: add that external set to training (multi-source)\n')
    D = '    d = df[df.split == split]\n'
    assert s.count(D) == 1, "ds() anchor not found"
    s = s.replace(D, '    d = df[df.split.isin(["train", "test_" + a.extra])] if (train and a.extra) else df[df.split == split]\n')
open("02_train.py", "w", encoding="utf-8").write(s)
print("patched; padaug:", 'a.strategy == "padaug"' in s, "| --extra:", '--extra' in s)
