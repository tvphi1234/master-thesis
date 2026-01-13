import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime

from utils import DEVICE
from models import CustomModel
from dataloader import get_dataloader, get_train_transforms, get_val_transforms


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
    parser.add_argument('--task-type', type=str, default='cancer',
                        choices=['cancer', 'stage', 'multi_task'],)
    return parser.parse_args()


# Parse command line arguments
args = parse_args()
os.makedirs(args.models_dir, exist_ok=True)


def calculate_loss(output, labels, levels, criterion):

    if args.task_type == 'cancer':
        loss = criterion(output, labels)
        num_samples = labels.size(0)

        # Use raw logits for argmax; values can be negative and that's fine
        predicted_output = output.argmax(dim=1)
        correct_output = (predicted_output == labels).sum().item()
    elif args.task_type == 'stage':
        loss = criterion(output, levels)
        num_samples = levels.size(0)

        predicted_output = output.argmax(dim=1)
        correct_output = (predicted_output == levels).sum().item()

    return {"loss": loss,
            "correct": correct_output,
            "num_samples": num_samples}


def run_epoch(model, data_loader, criterion, optimizer=None, training=False):

    total_dict = {
        "loss": 0.0,
        "correct": 0,
        "num_samples": 0
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

        output = model(inputs)

        loss_dict = calculate_loss(output, labels, levels, criterion)

        if training:
            loss_dict["loss"].backward()
            optimizer.step()

        total_dict["loss"] += loss_dict["loss"].item()
        total_dict["correct"] += loss_dict["correct"]
        total_dict["num_samples"] += loss_dict["num_samples"]

    average_loss = total_dict["loss"] / len(data_loader)
    accuracy = total_dict["correct"] / total_dict["num_samples"]
    return average_loss, accuracy


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
    }
    val_metrics = {
        "accuracies": [],
        "losses": [],
    }

    print(f"Starting training for {epochs} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Early stopping patience: {patience} epochs")
    print("-" * 60)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):

        # training
        model.train()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, training=True)

        train_metrics["accuracies"].append(train_accuracy)
        train_metrics["losses"].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion)

        val_metrics["accuracies"].append(val_accuracy)
        val_metrics["losses"].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | "
              f"Val   Acc: {val_accuracy:.4f}")

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
                                  annotation_files=['x4_train_annotations.csv', 
                                                    'x10_train_annotations.csv', 
                                                    'x10_warwick_train_annotations.csv',
                                                    'x40_train_annotations.csv'],
                                  data_transform=train_transform,
                                  is_shuffle=True,
                                  batch_size=args.batch_size,
                                  task_type=args.task_type)

    val_transform = get_val_transforms()
    val_loader = get_dataloader(args.data_dir,
                                annotation_files=['x4_val_annotations.csv',
                                                  'x10_val_annotations.csv', 
                                                  'x40_val_annotations.csv'],
                                data_transform=val_transform,
                                is_shuffle=False,
                                batch_size=1,
                                task_type=args.task_type)

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
