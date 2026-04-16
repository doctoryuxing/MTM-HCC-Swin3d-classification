import argparse
import json
import logging
import os
import pickle
import warnings
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.optim import lr_scheduler

from .config import TrainingConfig
from .data import get_transforms, load_dataset
from .losses import FocalLoss
from .model import build_model, count_parameters, get_param_groups


warnings.filterwarnings("ignore")


def setup_logger(log_path: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("swin3d")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def compute_metrics(labels: list[int], predictions: list[int], probabilities: list[np.ndarray], num_classes: int):
    labels_np = np.array(labels)
    predictions_np = np.array(predictions)
    probabilities_np = np.array(probabilities)

    if num_classes == 2:
        f1 = f1_score(labels_np, predictions_np, average="binary", zero_division=0)
        auc = 0.0
        if len(np.unique(labels_np)) > 1:
            auc = roc_auc_score(labels_np, probabilities_np[:, 1])
    else:
        f1 = f1_score(labels_np, predictions_np, average="macro", zero_division=0)
        auc = 0.0
        unique_labels = np.unique(labels_np)
        if len(unique_labels) > 1 and len(unique_labels) == probabilities_np.shape[1]:
            auc = roc_auc_score(labels_np, probabilities_np, multi_class="ovr", average="macro")

    return f1, auc


def build_result_row(epoch, batch_idx, file_name, true_label, pred_label, prob):
    row = {
        "epoch": epoch,
        "batch": batch_idx + 1,
        "file_name": file_name,
        "true_label": true_label,
        "predicted_label": pred_label,
        "correct": true_label == pred_label,
    }
    for class_index, class_probability in enumerate(prob):
        row[f"probability_class_{class_index}"] = class_probability
    return row


def get_class_names(num_classes: int) -> list[str]:
    return [f"Class {index}" for index in range(num_classes)]


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_classes: int):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_file_names = []

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device).expand(-1, 3, -1, -1, -1)
        labels = batch["label"].long().to(device)
        file_names = batch["file_name"]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            batch_correct = (predictions == labels).sum().item()

            correct += batch_correct
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)

            for index in range(len(labels)):
                true_label = labels[index].cpu().item()
                pred_label = predictions[index].cpu().item()
                prob = probabilities[index].cpu().numpy()
                file_name = file_names[index]

                row = build_result_row(epoch + 1, batch_idx, file_name, true_label, pred_label, prob)
                row["loss"] = loss.item()
                batch_results.append(row)

                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_probabilities.append(prob)
                all_file_names.append(file_name)

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    f1, auc = compute_metrics(all_labels, all_predictions, all_probabilities, num_classes=num_classes)
    return (
        epoch_loss,
        epoch_acc,
        auc,
        f1,
        batch_results,
        all_predictions,
        all_labels,
        all_probabilities,
        all_file_names,
    )


def evaluate(model, data_loader, criterion, device, num_classes: int, epoch=None):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_file_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].to(device).expand(-1, 3, -1, -1, -1)
            labels = batch["label"].long().to(device)
            file_names = batch["file_name"]

            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for index in range(len(labels)):
                true_label = labels[index].cpu().item()
                pred_label = predicted[index].cpu().item()
                prob = probabilities[index].cpu().numpy()
                file_name = file_names[index]

                results.append(
                    build_result_row(
                        epoch + 1 if epoch is not None else "final",
                        batch_idx,
                        file_name,
                        true_label,
                        pred_label,
                        prob,
                    )
                )

                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_probabilities.append(prob)
                all_file_names.append(file_name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    f1, auc = compute_metrics(all_labels, all_predictions, all_probabilities, num_classes=num_classes)
    return (
        epoch_loss,
        accuracy,
        auc,
        f1,
        results,
        all_predictions,
        all_labels,
        all_probabilities,
        all_file_names,
    )


def save_prediction_excel(best_results, final_results, output_path):
    best_df = pd.DataFrame(best_results)
    final_df = pd.DataFrame(final_results)
    probability_columns = sorted(
        [column for column in best_df.columns if column.startswith("probability_class_")],
        key=lambda column_name: int(column_name.rsplit("_", maxsplit=1)[-1]),
    )
    columns_to_keep = ["file_name", "true_label", "predicted_label", *probability_columns, "correct"]
    best_df = best_df[columns_to_keep]
    final_df = final_df[columns_to_keep]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        best_df.to_excel(writer, sheet_name="Best_Model", index=False)
        final_df.to_excel(writer, sheet_name="Final_Model", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train swin3d_s for 3D medical image classification.")
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--train-dir", default="data/train/ap")
    data_group.add_argument("--val-dir", default="data/val/ap")
    data_group.add_argument("--train-excel", default="data/labels/train.xlsx")
    data_group.add_argument("--val-excel", default="data/labels/val.xlsx")
    data_group.add_argument("--name-column", default="name")
    data_group.add_argument("--label-column", default="MTM")
    data_group.add_argument("--filename-delimiter", default="-")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--backbone", choices=["swin3d_s", "r3d_18"], default="swin3d_s",
                             dest="backbone_name", help="Backbone architecture (default: swin3d_s)")
    model_group.add_argument("--num-classes", type=int, default=2)
    model_group.add_argument("--resize-x", type=int, default=64)
    model_group.add_argument("--resize-y", type=int, default=64)
    model_group.add_argument("--resize-z", type=int, default=16)
    model_group.add_argument("--intensity-min", type=float, default=0.0)
    model_group.add_argument("--intensity-max", type=float, default=2000.0)
    model_group.add_argument("--no-finetune-last-stage", dest="finetune_last_stage", action="store_false")
    model_group.add_argument("--finetune-last-stage", dest="finetune_last_stage", action="store_true")
    model_group.add_argument("--backbone-lr", type=float, default=1e-5)
    model_group.add_argument("--weight-decay", type=float, default=1e-4)

    augmentation_group = parser.add_argument_group("augmentation")
    augmentation_group.add_argument("--simple-augmentation", action="store_true")

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--epochs", type=int, default=500)
    optimization_group.add_argument("--batch-size", type=int, default=16)
    optimization_group.add_argument("--learning-rate", type=float, default=1e-4)
    optimization_group.add_argument("--num-workers", type=int, default=4)
    optimization_group.add_argument("--step-size", type=int, default=7)
    optimization_group.add_argument("--gamma", type=float, default=0.1)
    optimization_group.set_defaults(finetune_last_stage=True)

    loss_group = parser.add_argument_group("loss")
    loss_group.add_argument("--no-focal-loss", action="store_true")
    loss_group.add_argument("--focal-gamma", type=float, default=3.0)

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--results-dir", default="outputs")
    output_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    args = parser.parse_args()

    config = TrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_excel=args.train_excel,
        val_excel=args.val_excel,
        name_column=args.name_column,
        label_column=args.label_column,
        backbone_name=args.backbone_name,
        num_classes=args.num_classes,
        resized_shape=(args.resize_x, args.resize_y, args.resize_z),
        intensity_range=(args.intensity_min, args.intensity_max),
        use_advanced_augmentation=not args.simple_augmentation,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        filename_delimiter=args.filename_delimiter,
        step_size=args.step_size,
        gamma=args.gamma,
        use_focal_loss=not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
        finetune_last_stage=args.finetune_last_stage,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        results_dir=args.results_dir,
    )
    return config, args.log_level


def main():
    config, log_level = parse_args()

    if config.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if config.kmp_duplicate_lib_ok:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    os.makedirs(config.results_dir, exist_ok=True)
    logger = setup_logger(config.log_path, log_level=log_level)

    with open(config.config_snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("3D Medical Image Classification - %s", config.backbone_name)
    logger.info("=" * 80)
    logger.info("Backbone: %s", config.backbone_name)
    logger.info("Image size: %s", config.resized_shape)
    logger.info("Batch size: %s", config.batch_size)
    logger.info("Head LR: %s", config.learning_rate)
    logger.info("Backbone LR: %s", config.backbone_lr)
    logger.info("Weight decay: %s", config.weight_decay)
    logger.info("Number of epochs: %s", config.num_epochs)
    logger.info("Finetune last stage: %s", config.finetune_last_stage)
    logger.info("Device: %s", config.device)
    logger.info("Results directory: %s", config.results_dir)
    logger.info("Log file: %s", config.log_path)
    logger.info("Config snapshot: %s", config.config_snapshot_path)
    logger.info("=" * 80)

    if config.use_focal_loss and len(config.focal_alpha) != config.num_classes:
        raise ValueError(
            "Length of focal_alpha must match num_classes. "
            f"Received len(focal_alpha)={len(config.focal_alpha)} and num_classes={config.num_classes}."
        )

    train_files = load_dataset(
        config.train_dir,
        config.train_excel,
        config.label_column,
        config.name_column,
        filename_delimiter=config.filename_delimiter,
    )
    val_files = load_dataset(
        config.val_dir,
        config.val_excel,
        config.label_column,
        config.name_column,
        filename_delimiter=config.filename_delimiter,
    )
    logger.info("Train samples: %s", len(train_files))
    logger.info("Validation samples: %s", len(val_files))

    train_transforms, val_transforms = get_transforms(config)
    train_dataset = Dataset(data=train_files, transform=train_transforms)
    val_dataset = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    logger.info("Building model...")
    model = build_model(config).to(config.device)
    param_stats = count_parameters(model)
    logger.info("Model loaded to %s", config.device)
    logger.info("Total params: %s", f"{param_stats['total']:,}")
    logger.info("Trainable params: %s", f"{param_stats['trainable']:,}")
    logger.info("Frozen params: %s", f"{param_stats['frozen']:,}")
    if config.finetune_last_stage:
        if config.backbone_name == "r3d_18":
            logger.info("Fine-tuning mode: layer4 + Head")
        else:
            logger.info("Fine-tuning mode: last Stage + PatchMerging + Norm + Head")
    else:
        logger.info("Fine-tuning mode: Head only (linear probing)")

    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        logger.info("Using Focal Loss (gamma=%s, alpha=%s)", config.focal_gamma, config.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss")

    param_groups = get_param_groups(model, config)
    optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    best_accuracy = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(config.num_epochs):
        logger.info("Epoch [%s/%s]", epoch + 1, config.num_epochs)
        train_loss, train_acc, train_auc, train_f1, *_ = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch, config.num_classes
        )
        logger.info(
            "Train - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1-Score: %.4f",
            train_loss,
            train_acc / 100,
            train_auc,
            train_f1,
        )

        val_loss, val_acc, val_auc, val_f1, *_ = evaluate(
            model, val_loader, criterion, config.device, config.num_classes, epoch
        )
        logger.info(
            "Val   - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1-Score: %.4f",
            val_loss,
            val_acc / 100,
            val_auc,
            val_f1,
        )

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc / 100,
                "train_auc": train_auc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc / 100,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        if config.save_best_model and val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.best_model_path)
            logger.info("Best model saved with accuracy: %.4f", best_accuracy / 100)

    if config.save_best_model and best_epoch == 0:
        torch.save(model.state_dict(), config.best_model_path)
        best_epoch = config.num_epochs
        best_accuracy = val_acc

    torch.save(model.state_dict(), config.final_model_path)
    pd.DataFrame(training_history).to_csv(config.training_history_path, index=False)
    logger.info("Final model saved to: %s", config.final_model_path)
    logger.info("Training history saved to: %s", config.training_history_path)

    logger.info("Loading best model from epoch %s...", best_epoch)
    model.load_state_dict(torch.load(config.best_model_path))
    best_train = evaluate(model, train_loader, criterion, config.device, config.num_classes, epoch=None)
    best_val = evaluate(model, val_loader, criterion, config.device, config.num_classes, epoch=None)

    best_train_loss, best_train_acc, best_train_auc, best_train_f1, best_train_results, best_train_preds, best_train_labels, best_train_probs, best_train_files = best_train
    best_val_loss, best_val_acc, best_val_auc, best_val_f1, best_val_results, best_val_preds, best_val_labels, best_val_probs, best_val_files = best_val

    for result in best_train_results:
        result["model"] = "best"
        result["epoch"] = best_epoch
    for result in best_val_results:
        result["model"] = "best"
        result["epoch"] = best_epoch

    logger.info("Loading final model from epoch %s...", config.num_epochs)
    model.load_state_dict(torch.load(config.final_model_path))
    final_train = evaluate(model, train_loader, criterion, config.device, config.num_classes, epoch=None)
    final_val = evaluate(model, val_loader, criterion, config.device, config.num_classes, epoch=None)

    final_train_loss, final_train_acc, final_train_auc, final_train_f1, final_train_results, final_train_preds, final_train_labels, final_train_probs, final_train_files = final_train
    final_val_loss, final_val_acc, final_val_auc, final_val_f1, final_val_results, final_val_preds, final_val_labels, final_val_probs, final_val_files = final_val

    for result in final_train_results:
        result["model"] = "final"
        result["epoch"] = config.num_epochs
    for result in final_val_results:
        result["model"] = "final"
        result["epoch"] = config.num_epochs

    train_excel_path = config.train_results_path.replace(".csv", ".xlsx")
    val_excel_path = config.val_results_path.replace(".csv", ".xlsx")
    save_prediction_excel(best_train_results, final_train_results, train_excel_path)
    save_prediction_excel(best_val_results, final_val_results, val_excel_path)
    logger.info("Train predictions saved to: %s", train_excel_path)
    logger.info("Validation predictions saved to: %s", val_excel_path)

    predictions_data = {
        "best_model": {
            "epoch": best_epoch,
            "train": {
                "predictions": best_train_preds,
                "labels": best_train_labels,
                "probabilities": best_train_probs,
                "file_names": best_train_files,
                "metrics": {
                    "loss": best_train_loss,
                    "accuracy": best_train_acc / 100,
                    "auc": best_train_auc,
                    "f1": best_train_f1,
                },
            },
            "val": {
                "predictions": best_val_preds,
                "labels": best_val_labels,
                "probabilities": best_val_probs,
                "file_names": best_val_files,
                "metrics": {
                    "loss": best_val_loss,
                    "accuracy": best_val_acc / 100,
                    "auc": best_val_auc,
                    "f1": best_val_f1,
                },
            },
        },
        "final_model": {
            "epoch": config.num_epochs,
            "train": {
                "predictions": final_train_preds,
                "labels": final_train_labels,
                "probabilities": final_train_probs,
                "file_names": final_train_files,
                "metrics": {
                    "loss": final_train_loss,
                    "accuracy": final_train_acc / 100,
                    "auc": final_train_auc,
                    "f1": final_train_f1,
                },
            },
            "val": {
                "predictions": final_val_preds,
                "labels": final_val_labels,
                "probabilities": final_val_probs,
                "file_names": final_val_files,
                "metrics": {
                    "loss": final_val_loss,
                    "accuracy": final_val_acc / 100,
                    "auc": final_val_auc,
                    "f1": final_val_f1,
                },
            },
        },
        "config": {
            "backbone_name": config.backbone_name,
            "num_classes": config.num_classes,
            "resized_shape": config.resized_shape,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma,
            "focal_alpha": config.focal_alpha,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(config.predictions_path, "wb") as handle:
        pickle.dump(predictions_data, handle)
    logger.info("Predictions pickle saved to: %s", config.predictions_path)

    logger.info("%s", "=" * 80)
    logger.info("Final Evaluation Report")
    logger.info("%s", "=" * 80)
    logger.info("[Best Model] (Epoch %s)", best_epoch)
    logger.info(
        "Train Set - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1: %.4f",
        best_train_loss,
        best_train_acc / 100,
        best_train_auc,
        best_train_f1,
    )
    logger.info(
        "Val Set   - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1: %.4f",
        best_val_loss,
        best_val_acc / 100,
        best_val_auc,
        best_val_f1,
    )
    class_names = get_class_names(config.num_classes)
    logger.info(
        "Best Model - Train Set Classification Report:\n%s",
        classification_report(best_train_labels, best_train_preds, labels=list(range(config.num_classes)), target_names=class_names, zero_division=0),
    )
    logger.info(
        "Best Model - Val Set Classification Report:\n%s",
        classification_report(best_val_labels, best_val_preds, labels=list(range(config.num_classes)), target_names=class_names, zero_division=0),
    )
    logger.info("[Final Model] (Epoch %s)", config.num_epochs)
    logger.info(
        "Train Set - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1: %.4f",
        final_train_loss,
        final_train_acc / 100,
        final_train_auc,
        final_train_f1,
    )
    logger.info(
        "Val Set   - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1: %.4f",
        final_val_loss,
        final_val_acc / 100,
        final_val_auc,
        final_val_f1,
    )
    logger.info(
        "Final Model - Train Set Classification Report:\n%s",
        classification_report(final_train_labels, final_train_preds, labels=list(range(config.num_classes)), target_names=class_names, zero_division=0),
    )
    logger.info(
        "Final Model - Val Set Classification Report:\n%s",
        classification_report(final_val_labels, final_val_preds, labels=list(range(config.num_classes)), target_names=class_names, zero_division=0),
    )
    logger.info("%s", "=" * 80)
    logger.info("Training completed!")
    logger.info("Results saved in: %s", config.results_dir)
    logger.info("Best model: %s (Epoch %s, Acc: %.4f)", config.best_model_path, best_epoch, best_accuracy / 100)
    logger.info("Final model: %s (Epoch %s)", config.final_model_path, config.num_epochs)
    logger.info("%s", "=" * 80)
