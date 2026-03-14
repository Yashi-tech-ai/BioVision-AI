# BIOVISION-AI

**AI-based skin lesion and skin cancer detection** — a production-ready academic project for dermatology decision-support using computer vision and deep learning.

## Overview

BIOVISION-AI is an end-to-end system that:

1. Accepts a skin lesion (dermoscopic) image
2. Preprocesses and optionally segments the lesion
3. Classifies the lesion type and risk
4. Explains the prediction with Grad-CAM heatmaps
5. Exposes an API and a simple web UI for demonstration

**Scope:** Dermatology only (skin lesion detection and classification). Not a replacement for clinical diagnosis.

## Requirements

- Python 3.10+
- PyTorch, torchvision, timm, OpenCV, Albumentations, grad-cam, FastAPI, Streamlit (see `requirements.txt`)

## Installation

```bash
cd /path/to/PBL_CURSOR
pip install -r requirements.txt
```

## Dataset setup

### Option A: Dummy data (quick start)

```bash
python scripts/create_dummy_data.py --data_root ./data --num_samples 200
```

### Option B: HAM10000

1. Get [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T): download `HAM10000_metadata.csv` and image zip.
2. Unzip images into `data/ham10000_images/` or `data/images/`.
3. Put `HAM10000_metadata.csv` in `data/`.
4. CSV must have columns `image_id` and `dx` (values: akiec, bcc, bkl, df, mel, nv, vasc).

### Option C: ISIC

1. Download from [ISIC Archive](https://www.isic-archive.com/).
2. Place metadata CSV in `data/` (e.g. `metadata.csv`) with `image_id` and `dx`.
3. Place images in `data/images/` with filenames matching `image_id`.

### Expected structure

```
data/
  metadata.csv          # or HAM10000_metadata.csv
  images/
    *.jpg / *.png
  [optional] masks/     # same base name as image, for segmentation
```

## Training

```bash
# Classification (EfficientNet-B0, focal loss, class weights)
python scripts/train.py --config configs/default.yaml --data_root ./data

# Optional: override data root
python scripts/train.py --config configs/classification.yaml --data_root ./data
```

Checkpoints are saved to `checkpoints/classification/best.pt` (configurable in YAML). Training uses early stopping and cosine LR schedule.

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/classification/best.pt --config configs/default.yaml --output_dir evaluation_outputs
```

This produces:

- Metrics: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix plot
- ROC curves
- Precision-recall curves
- Classification report

## Inference (CLI)

```bash
python scripts/predict.py --checkpoint checkpoints/classification/best.pt --image path/to/lesion.jpg --output_dir predict_outputs
```

Grad-CAM heatmap and overlay are saved to `output_dir` unless `--no_gradcam` is set.

## API

```bash
# Start server (set checkpoint via env or default path)
python api/run_api.py --checkpoint checkpoints/classification/best.pt --port 8000
```

- **GET /health** — Health check
- **POST /predict** — Upload image (multipart), optional form fields: age, sex, body_location  
  - Response: `predicted_class`, `confidence`, `risk_category`, `probabilities`, `heatmap_path`, `overlay_path`

Example:

```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:8000/predict
```

## Demo UI (Streamlit)

```bash
streamlit run frontend/app.py
```

Then open the URL (e.g. http://localhost:8501). You can:

- Upload an image
- See predicted class, confidence, risk category
- See class probabilities
- See Grad-CAM overlay

## Project structure

```
biovision-ai/
  configs/           # YAML configs
  data/
    datasets/       # SkinLesionDataset, get_dataloaders
    preprocessing.py
    augmentation.py
  models/
    segmentation/   # U-Net
    classification/ # EfficientNet-B0 classifier
  training/         # Losses (Dice+BCE, Focal)
  evaluation/       # Metrics, confusion matrix, ROC, PR
  explainability/   # Grad-CAM
  api/              # FastAPI app
  frontend/         # Streamlit app
  utils/
  scripts/          # train, evaluate, predict, create_dummy_data
  notebooks/        # (optional) Jupyter notebooks
  README.md
  requirements.txt
```

## Model architecture

- **Classification:** EfficientNet-B0 (timm), pretrained on ImageNet, head: dropout + linear → 7 classes.
- **Segmentation (optional):** U-Net, Dice + BCE loss, for datasets that provide masks.
- **Explainability:** Grad-CAM on the classifier backbone, integrated in predict script, API, and Streamlit.

## Configuration

Edit `configs/default.yaml` for:

- `data.root`, `data.batch_size`, `data.image_size`
- `classification.epochs`, `classification.lr`, `classification.loss` (focal / cross_entropy)
- `classification.early_stopping_patience`
- API/frontend ports

## Reproducibility

- Seed is set in config (`seed: 42`); used for train/val/test split.
- Checkpoints save `config` and `epoch`; evaluation and inference use the same config when provided.

## Disclaimer

BIOVISION-AI is for **educational and decision-support** use only. It does not replace clinical diagnosis, biopsy, or histopathology. Always follow institutional and regulatory guidelines.

## License

Use for academic and educational purposes.
