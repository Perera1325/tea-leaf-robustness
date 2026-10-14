"""RQ3: float32 vs float16 vs int8 TFLite. Calibrate int8 with ~200 TRAIN images (representative dataset).
usage: python 04_export_tflite.py model.keras   -> model_fp32/fp16/int8.tflite
"""
import sys, pandas as pd, tensorflow as tf
m = tf.keras.models.load_model(sys.argv[1]); stem = sys.argv[1].replace(".keras", "")
df = pd.read_csv("splits.csv"); rep = df[df.split == "train"].sample(min(200, (df.split == "train").sum()), random_state=0).path.values
def rep_ds():
    for p in rep:
        x = tf.image.resize(tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False), (224, 224))
        yield [tf.expand_dims(tf.cast(x, tf.float32), 0)]
def conv(name, setup):
    c = tf.lite.TFLiteConverter.from_keras_model(m); setup(c)
    open(f"{stem}_{name}.tflite", "wb").write(c.convert()); print(name, "ok")
conv("fp32", lambda c: None)
conv("fp16", lambda c: (setattr(c, "optimizations", [tf.lite.Optimize.DEFAULT]), setattr(c.target_spec, "supported_types", [tf.float16])))
def i8(c):
    c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep_ds
conv("int8", i8)  # dynamic-range int8 with float IO
