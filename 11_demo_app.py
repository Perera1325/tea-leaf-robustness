"""Offline demo: tea leaf disease check with a 'retake photo' reject option.
Runs the fp16 TFLite model locally (no internet). Usage: python 11_demo_app.py
Research prototype - NOT a diagnosis tool.
"""
import numpy as np
from PIL import Image
import gradio as gr

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

MODEL = "models/mobilenetv3_baseline_s42_ft1e4_fp16.tflite"
CLASSES = ["Algal_leaf_spot", "Black_blight", "Blister_blight", "Gray_blight", "Healthy", "Spider_mites"]  # = classes.txt order
T, TAU = 1.366, 0.791   # temperature and reject threshold, both fitted on the validation split only

interp = Interpreter(model_path=MODEL, num_threads=2)
interp.allocate_tensors()
I_IN = interp.get_input_details()[0]["index"]
I_OUT = interp.get_output_details()[0]["index"]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict(img):
    if img is None:
        return {}, "Upload or take a photo of one tea leaf."
    x = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)[None]  # 0-255, model has its own preprocessing
    interp.set_tensor(I_IN, x)
    interp.invoke()
    p = softmax(interp.get_tensor(I_OUT)[0] / T)
    top = int(p.argmax())
    probs = {c: float(v) for c, v in zip(CLASSES, p)}
    if p[top] >= TAU:
        msg = f"**Likely: {CLASSES[top]}** (confidence {p[top]:.0%})\n\nThis is a research prototype, not a diagnosis."
    else:
        msg = (f"**Not sure - please retake the photo.** (confidence {p[top]:.0%} is below the {TAU:.0%} rule)\n\n"
               "Tips: one leaf filling most of the frame, good daylight, in focus.\n\n"
               "This is a research prototype, not a diagnosis.")
    return probs, msg


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Tea leaf photo"),
    outputs=[gr.Label(num_top_classes=3, label="Top predictions"), gr.Markdown()],
    title="Tea leaf disease check (offline prototype)",
    description=("Final-year research prototype. Trained on one public dataset; accuracy drops sharply on photos taken in "
                 "different conditions (see the report). When unsure it asks you to retake the photo."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
