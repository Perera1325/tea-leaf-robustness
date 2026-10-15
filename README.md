# Tea Leaf Disease Detection: Cross-Dataset Robustness and Offline Deployment

Final-year project (BSc Computer Science, University of Bedfordshire). Author: Vinod Ravindu Perera.

Goal: measure how tea leaf disease classifiers generalise to unseen public datasets, test ways to
reduce the drop, add calibrated "not sure" predictions, and compare on-device (TFLite) vs full model.

Data: public datasets only (not stored in this repo).

Status: work in progress, no results claimed yet.

## Offline demo (research prototype, not a diagnosis tool)

Runs the fp16 TFLite model (`models/mobilenetv3_baseline_s42_ft1e4_fp16.tflite`) locally, no internet needed.
When the calibrated confidence is below 0.791 it asks the user to retake the photo
(temperature T=1.366 and threshold tau=0.791 were fitted on the validation split only).

    python -m venv .venv
    source .venv/Scripts/activate      # Windows Git Bash
    python -m pip install -r requirements.txt
    python 11_demo_app.py              # then open http://127.0.0.1:7860

Known limits: accuracy falls sharply on photos from other sources (see results/), and the app has no
"not a leaf" class, so a confident answer on a non-leaf photo is possible.
