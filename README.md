# Swin3D Medical Classification

PyTorch and MONAI project for single-sequence 3D medical image classification with `swin3d_s`.

## Features

- Fixed backbone: `torchvision.models.video.swin3d_s`
- Single-sequence 3D classification
- Optional `FocalLoss` for class imbalance
- Training / validation metrics: loss, accuracy, AUC, F1
- Best-model and final-model evaluation export
- Excel and pickle result export

## Project Structure

```text
swin3d-github/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── train.py
└── swin3d/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── losses.py
    ├── model.py
    └── train.py
```

## Installation

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install -e .
```

## Data Layout

Expected directory layout:

```text
data/
├── train/
│   └── ap/
│       ├── case001-AP.nii.gz
│       └── ...
├── val/
│   └── ap/
│       ├── case101-AP.nii.gz
│       └── ...
└── labels/
    ├── train.xlsx
    └── val.xlsx
```

The Excel files must contain:

- one case identifier column, for example `name`
- one label column, for example `MTM`

The current matching rule extracts the case id from the image filename before the first `-`.

Example:

- image file: `case001-AP.nii.gz`
- matched case id: `case001`

## Usage

Run with defaults:

```bash
python train.py
```

Run with custom paths:

```bash
python train.py \
  --train-dir /path/to/train/ap \
  --val-dir /path/to/val/ap \
  --train-excel /path/to/train.xlsx \
  --val-excel /path/to/val.xlsx \
  --name-column name \
  --label-column MTM \
  --results-dir /path/to/results
```

Show more verbose logs:

```bash
python train.py --log-level DEBUG
```

Disable focal loss:

```bash
python train.py --no-focal-loss
```

Use simple augmentation:

```bash
python train.py --simple-augmentation
```

## Main Outputs

The training script writes:

- `best_model.pth`
- `final_model.pth`
- `training_history.csv`
- `train_results.xlsx`
- `val_results.xlsx`
- `predictions.pkl`
- `training.log`
- `config_snapshot.json`

## Notes

- Input images are resized to `64 x 64 x 16` by default.
- The script expands single-channel volumes to 3 channels before feeding `swin3d_s`.
- Feature extractor parameters are frozen by default; only the classification head is optimized.
