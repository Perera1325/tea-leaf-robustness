# Datasets

Images are NOT stored in this repository. Each dataset keeps its own licence.

## Source domain (train / val / in-domain test)
- Sri Lankan Tea Leaf Dataset, Mendeley Data `sjmy6k24d6` (v5), licence CC BY 4.0.
- Raw images only (1,791 images, 6 classes: Algal_leaf_spot, Black_blight, Blister_blight, Gray_blight, Healthy, Spider_mites).
  The augmented copies are not used.

## External test domain 1: Hugging Face `yunusserhat/tea_sickness_dataset`
- 885 images (train 708 / validation 88 / test 89), 8 classes (checked 2026-10-14).
- Licence and origin are NOT stated on the dataset page. Used only as an external test set, not redistributed.
- Only overlapping classes are used: algal_leaf -> Algal_leaf_spot, gray_light -> Gray_blight, healthy -> Healthy.
  Other classes (Anthracnose, bird_eye_spot, brown_blight, red_leaf_spot, white_spot) are dropped.
- Fetch with `python 05_fetch_hf_tea.py` (writes `data/hf_tea/` and `manifests/hf_tea_manifest.csv`).
- Near-duplicate check against the source dataset: to be recorded here.

## Planned
- Mendeley `mkzyfj8bkj` (raw images only; Red Spider -> Spider_mites, Normal -> Healthy).
