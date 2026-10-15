"""Train MobileNetV3 / EfficientNet-Lite-style baselines + robustness strategies.
usage: python 02_train.py --arch mobilenetv3 --strategy baseline|fieldaug|classweight|dann_free
Strategies (RQ2): baseline | fieldaug (blur, noise, brightness, JPEG, random-erasing) | classweight
"""
import argparse, pandas as pd, numpy as np, tensorflow as tf
ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="mobilenetv3")
ap.add_argument("--strategy", default="baseline")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ft_lr", type=float, default=1e-5)
ap.add_argument("--tag", default="")
ap.add_argument("--extra", default="")   # e.g. hf_tea or mendeley_ext: add that external set to training (multi-source)
a = ap.parse_args()
tf.keras.utils.set_random_seed(a.seed)
df = pd.read_csv("splits.csv")
classes = sorted(df.cls.unique()); idx = {c: i for i, c in enumerate(classes)}
IMG = 224

def load(path, y, train):
    x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    x = tf.image.resize(x, (IMG, IMG))
    if train:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        if a.strategy == "padaug":     # leaf shrunk onto a plain light background
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
        if a.strategy == "fieldaug":   # simulate phone/field shift
            x = tf.image.random_brightness(x, 40.0)
            x = tf.image.random_contrast(x, 0.6, 1.4)
            x = tf.image.random_saturation(x, 0.6, 1.4)
            x = x + tf.random.normal(tf.shape(x), 0, 8.0)
            x = tf.cast(tf.image.random_jpeg_quality(tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8), 40, 95), tf.float32)
    return x, y

def ds(split, train=False):
    d = df[df.split.isin(["train", "test_" + a.extra])] if (train and a.extra) else df[df.split == split]
    t = tf.data.Dataset.from_tensor_slices((d.path.values, d.cls.map(idx).values))
    if train: t = t.shuffle(4096, seed=a.seed)
    return t.map(lambda p, y: load(p, y, train), num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(2)

base = {"mobilenetv3": tf.keras.applications.MobileNetV3Large,
        "efficientnet": tf.keras.applications.EfficientNetB0}[a.arch](include_top=False, weights="imagenet",
        input_shape=(IMG, IMG, 3), pooling="avg")  # models include their own preprocessing (0-255 input)
base.trainable = False
m = tf.keras.Sequential([base, tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(classes))])  # logits (needed for temperature scaling)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
cw = None
if a.strategy == "classweight":
    cnt = df[df.split == "train"].cls.value_counts()
    cw = {idx[c]: len(df[df.split == "train"]) / (len(cnt) * n) for c, n in cnt.items()}
cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
m.compile(tf.keras.optimizers.Adam(1e-3), loss, metrics=["accuracy"])
m.fit(ds("train", True), validation_data=ds("val"), epochs=5, class_weight=cw, callbacks=cb)
base.trainable = True
m.compile(tf.keras.optimizers.Adam(a.ft_lr), loss, metrics=["accuracy"])
m.fit(ds("train", True), validation_data=ds("val"), epochs=a.epochs, class_weight=cw, callbacks=cb)
name = f"{a.arch}_{a.strategy}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
m.save(f"{name}.keras")
open("classes.txt", "w").write("\n".join(classes))
print("saved", name)
