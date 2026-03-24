import os

import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
)

from .config import TrainingConfig


def strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def infer_case_id_candidates(filename: str, delimiter: str) -> list[str]:
    candidates = [filename]
    if delimiter and delimiter in filename:
        prefix = filename.split(delimiter)[0]
        if prefix not in candidates:
            candidates.append(prefix)
    return candidates


def get_transforms(config: TrainingConfig):
    common = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=config.resized_shape),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.intensity_range[0],
            a_max=config.intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if config.use_advanced_augmentation:
        train_transforms = Compose(
            common
            + [
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=[0, 1]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                EnsureTyped(keys=["image"]),
            ]
        )
    else:
        train_transforms = Compose(
            common
            + [
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=[0, 1]),
                EnsureTyped(keys=["image"]),
            ]
        )

    val_transforms = Compose(common + [EnsureTyped(keys=["image"])])
    return train_transforms, val_transforms


def load_dataset(
    data_dir: str,
    label_excel: str,
    label_column: str,
    name_column: str,
    filename_delimiter: str = "-",
) -> list[dict]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not os.path.isfile(label_excel):
        raise FileNotFoundError(f"Label file does not exist: {label_excel}")

    df = pd.read_excel(label_excel)
    required_columns = {name_column, label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns in label file: {sorted(missing_columns)}")

    df = df.copy()
    df[name_column] = df[name_column].astype(str).str.strip()
    duplicated_ids = df[df[name_column].duplicated()][name_column].unique().tolist()
    if duplicated_ids:
        raise ValueError(f"Duplicate case ids found in label file: {duplicated_ids[:10]}")

    label_lookup = dict(zip(df[name_column], df[label_column]))
    image_files = [
        os.path.join(data_dir, file_name)
        for file_name in sorted(os.listdir(data_dir))
        if file_name.endswith(".nii.gz") or file_name.endswith(".nii")
    ]

    data = []
    unmatched_files = []
    duplicate_image_case_ids = []
    seen_case_ids = set()
    for image_path in image_files:
        base_name = os.path.basename(image_path)
        file_name = strip_nii_suffix(base_name)
        matched_case_id = None
        for candidate in infer_case_id_candidates(file_name, delimiter=filename_delimiter):
            if candidate in label_lookup:
                matched_case_id = candidate
                break

        if matched_case_id is None:
            unmatched_files.append(base_name)
            continue

        if matched_case_id in seen_case_ids:
            duplicate_image_case_ids.append(matched_case_id)
            continue

        seen_case_ids.add(matched_case_id)
        data.append(
            {
                "image": image_path,
                "label": label_lookup[matched_case_id],
                "file_name": matched_case_id,
            }
        )

    if unmatched_files:
        raise ValueError(
            "Found image files without matching labels. "
            f"Examples: {unmatched_files[:10]}"
        )
    if duplicate_image_case_ids:
        raise ValueError(
            "Found duplicated case ids from image files after parsing. "
            f"Examples: {duplicate_image_case_ids[:10]}"
        )
    missing_image_labels = sorted(set(label_lookup) - seen_case_ids)
    if missing_image_labels:
        raise ValueError(
            "Found labels without matching image files. "
            f"Examples: {missing_image_labels[:10]}"
        )

    return data
