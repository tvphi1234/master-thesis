import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

import logging

from tqdm import tqdm
from datetime import datetime

from utils import DEVICE
from models import CustomModel
from dataloader import get_dataloader, get_train_transforms, get_val_transforms


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/training.log'),  # Save to file
        logging.StreamHandler()                # Also print to console
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the training data (default: data)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet50', 'repvgg_a0', 'repvgg_a1', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b2', 'repvgg_b3',
                                 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv3_large_100', 'mobilenetv3_small_100'],
                        help='Model architecture to use (default: xception)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for early stopping (default: 7)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models (default: models)')
    parser.add_argument('--task-type', type=str, default='multi_task',
                        choices=['cancer', 'stage', 'multi_task'],)
    return parser.parse_args()


# Parse command line arguments
args = parse_args()
os.makedirs(args.models_dir, exist_ok=True)


def calculate_loss(cancer_output, level_output, labels, levels, criterion):
    loss_cancer = criterion(cancer_output, labels)
    num_samples_cancer = labels.size(0)

    # Use raw logits for argmax; values can be negative and that's fine
    predicted_cancer = cancer_output.argmax(dim=1)
    correct_cancer = (predicted_cancer == labels).sum().item()

    # skip level loss for entries where level <= 0 (e.g. -1 or 0)
    # levels: tensor of shape (batch,), with non-positive values meaning 'no label'
    valid_mask = levels >= 0

    if valid_mask.any():
        # select only valid samples
        # If your level labels are 1..K, convert to 0..K-1 for CrossEntropyLoss
        level_targets = levels[valid_mask]
        level_preds = level_output[valid_mask]

        loss_level = criterion(level_preds, level_targets)
        num_samples_level = level_targets.size(0)

        predicted_levels = torch.argmax(level_preds, 1)
        correct_level = (predicted_levels == level_targets).sum().item()
    else:
        # no valid level labels in this batch
        loss_level = torch.tensor(0.0, device=DEVICE)
        num_samples_level = 0
        correct_level = 0

    loss = (loss_cancer*num_samples_cancer + loss_level *
            num_samples_level) / (num_samples_cancer + num_samples_level)

    return {"loss": loss, "loss_cancer": loss_cancer, "loss_level": loss_level,
            "correct_cancer": correct_cancer, "correct_level": correct_level,
            "num_samples_cancer": num_samples_cancer, "num_samples_level": num_samples_level}


def run_epoch(model, data_loader, criterion, optimizer=None, training=False):

    total_dict = {
        "loss": 0.0,
        "loss_cancer": 0.0,
        "loss_level": 0.0,
        "correct_cancer": 0,
        "correct_level": 0,
        "num_samples_cancer": 0,
        "num_samples_level": 0
    }

    if training:
        desc = "Training Batches "
    else:
        desc = "Validation Batches "

    for inputs, labels, levels in tqdm(data_loader, desc=desc):
        inputs, labels, levels = inputs.to(
            DEVICE), labels.to(DEVICE), levels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        cancer_output, level_output = model(inputs)

        loss_dict = calculate_loss(
            cancer_output, level_output, labels, levels, criterion)

        if training:
            loss_dict["loss"].backward()
            optimizer.step()

        total_dict["loss"] += loss_dict["loss"].item()
        total_dict["correct_cancer"] += loss_dict["correct_cancer"]
        total_dict["correct_level"] += loss_dict["correct_level"]
        total_dict["loss_cancer"] += loss_dict["loss_cancer"].item()
        total_dict["loss_level"] += loss_dict["loss_level"].item()
        total_dict["num_samples_cancer"] += loss_dict["num_samples_cancer"]
        total_dict["num_samples_level"] += loss_dict["num_samples_level"]

    average_loss = total_dict["loss"] / len(data_loader)
    average_cancer_loss = total_dict["loss_cancer"] / len(data_loader)
    average_level_loss = total_dict["loss_level"] / len(data_loader)
    accuracy = (total_dict["correct_cancer"] + total_dict["correct_level"]) / \
        (total_dict["num_samples_cancer"] + total_dict["num_samples_level"])
    accuracy_cancer = total_dict["correct_cancer"] / \
        total_dict["num_samples_cancer"]
    accuracy_level = total_dict["correct_level"] / \
        total_dict["num_samples_level"]
    return average_loss, average_cancer_loss, average_level_loss, accuracy, accuracy_cancer, accuracy_level


# Training Loop with best and last model saving, early stopping, and plotting
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_save_name, patience=7):
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # lists to store training accuracy and loss for plotting
    train_metrics = {
        "accuracies": [],
        "losses": [],
        "cancer_losses": [],
        "level_losses": [],
        "accuracies_cancer": [],
        "accuracies_level": []
    }
    val_metrics = {
        "accuracies": [],
        "losses": [],
        "cancer_losses": [],
        "level_losses": [],
        "accuracies_cancer": [],
        "accuracies_level": []
    }

    print(f"Starting training for {epochs} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Early stopping patience: {patience} epochs")
    print("-" * 60)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):

        # training
        model.train()
        train_loss, train_cancer_loss, train_level_loss, train_accuracy, train_accuracy_cancer, train_accuracy_level = run_epoch(
            model, train_loader, criterion, optimizer, training=True)

        train_metrics["accuracies"].append(train_accuracy)
        train_metrics["losses"].append(train_loss)
        train_metrics["cancer_losses"].append(train_cancer_loss)
        train_metrics["level_losses"].append(train_level_loss)
        train_metrics["accuracies_cancer"].append(train_accuracy_cancer)
        train_metrics["accuracies_level"].append(train_accuracy_level)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_cancer_loss, val_level_loss, val_accuracy, val_accuracy_cancer, val_accuracy_level = run_epoch(
                model, val_loader, criterion)

        val_metrics["accuracies"].append(val_accuracy)
        val_metrics["losses"].append(val_loss)
        val_metrics["cancer_losses"].append(val_cancer_loss)
        val_metrics["level_losses"].append(val_level_loss)
        val_metrics["accuracies_cancer"].append(val_accuracy_cancer)
        val_metrics["accuracies_level"].append(val_accuracy_level)

        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} (Cancer: {train_cancer_loss:.4f}, Level: {train_level_loss:.4f}) | "
              f"Train Acc: {train_accuracy:.4f} (Cancer: {train_accuracy_cancer:.4f}, Level: {train_accuracy_level:.4f})")
        logger.info(f"Val   Loss: {val_loss:.4f} (Cancer: {val_cancer_loss:.4f}, Level: {val_level_loss:.4f}) | "
              f"Val   Acc: {val_accuracy:.4f} (Cancer: {val_accuracy_cancer:.4f}, Level: {val_accuracy_level:.4f})")

        # Learning rate scheduler step
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")

        # Save best model and early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_train_accuracy = train_accuracy
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(
                f"✓ New best model! Validation accuracy: {best_val_accuracy:.4f}")

            # save best model immediately
            best_model_path = os.path.join(
                args.models_dir, f"{args.model}_best_{epoch+1}.pth")
            torch.save(best_model_state, best_model_path)
            print(
                f"Best model saved as {best_model_path} with accuracy: {best_val_accuracy:.4f}")
        elif val_accuracy == best_val_accuracy:
            if train_accuracy > max(train_metrics["accuracies"][:-1], default=0):
                best_model_state = model.state_dict().copy()
                print(
                    f"✓ Validation accuracy tied, but improved training accuracy: {train_accuracy:.4f}")
                epochs_without_improvement = 0
                best_model_path = os.path.join(
                    args.models_dir, f"{args.model}_best_{epoch+1}.pth")
                torch.save(best_model_state, best_model_path)
                print(
                    f"Best model saved as {best_model_path} with accuracy: {best_val_accuracy:.4f}")

            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement} epochs")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

        print("-" * 60)

    return model, best_model_state, best_val_accuracy, train_metrics["accuracies"], val_metrics["accuracies"], train_metrics["losses"], val_metrics["losses"]


if __name__ == "__main__":

    # model
    model = CustomModel(model_name=args.model,
                        task_type=args.task_type).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # get date with format yyyymmdd
    date_str = datetime.now().strftime("%Y%m%d")
    model_save_name = f"{args.model}_{date_str}"

    # Datasets and Dataloaders
    train_transform = get_train_transforms()
    train_loader = get_dataloader(args.data_dir,
                                  annotation_files=['x10_train_annotations.csv', 
                                                    'x10_warwick_train_annotations.csv',
                                                    'x40_train_annotations.csv'],
                                  data_transform=train_transform,
                                  is_shuffle=True,
                                  batch_size=args.batch_size,
                                  task_type=args.task_type
                                  )

    val_transform = get_val_transforms()
    val_loader = get_dataloader(args.data_dir,
                                annotation_files=['x10_val_annotations.csv', 
                                                  'x10_warwick_test_annotations.csv',
                                                  'x40_val_annotations.csv'],
                                data_transform=val_transform,
                                is_shuffle=False,
                                batch_size=1,
                                task_type=args.task_type
                                )

    # Train the model
    trained_model, best_model_state, best_accuracy, train_accs, val_accs, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, model_save_name, args.patience
    )

    # Save the models in the models directory
    last_model_path = os.path.join(
        args.models_dir, f"{model_save_name}_last.pth")
    best_model_path = os.path.join(
        args.models_dir, f"{model_save_name}_best.pth")

    torch.save(trained_model.state_dict(), last_model_path)
    print(f"Last model saved as {last_model_path}")

    torch.save(best_model_state, best_model_path)
    print(
        f"Best model saved as {best_model_path} with accuracy: {best_accuracy:.4f}")
