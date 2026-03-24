from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    train_dir: str = "data/train/ap"
    val_dir: str = "data/val/ap"
    train_excel: str = "data/labels/train.xlsx"
    val_excel: str = "data/labels/val.xlsx"
    name_column: str = "name"
    label_column: str = "MTM"
    backbone_name: str = "swin3d_s"
    num_classes: int = 2
    resized_shape: tuple[int, int, int] = (64, 64, 16)
    intensity_range: tuple[float, float] = (0.0, 2000.0)
    use_advanced_augmentation: bool = True
    num_epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 4
    filename_delimiter: str = "-"
    step_size: int = 7
    gamma: float = 0.1
    use_focal_loss: bool = True
    focal_gamma: float = 3.0
    focal_alpha: list[float] = field(default_factory=lambda: [0.3, 0.7])
    results_dir: str = "outputs"
    save_best_model: bool = True
    cuda_launch_blocking: bool = True
    kmp_duplicate_lib_ok: bool = True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def best_model_path(self) -> str:
        return str(Path(self.results_dir) / "best_model.pth")

    @property
    def final_model_path(self) -> str:
        return str(Path(self.results_dir) / "final_model.pth")

    @property
    def train_results_path(self) -> str:
        return str(Path(self.results_dir) / "train_results.csv")

    @property
    def val_results_path(self) -> str:
        return str(Path(self.results_dir) / "val_results.csv")

    @property
    def training_history_path(self) -> str:
        return str(Path(self.results_dir) / "training_history.csv")

    @property
    def predictions_path(self) -> str:
        return str(Path(self.results_dir) / "predictions.pkl")

    @property
    def log_path(self) -> str:
        return str(Path(self.results_dir) / "training.log")

    @property
    def config_snapshot_path(self) -> str:
        return str(Path(self.results_dir) / "config_snapshot.json")
