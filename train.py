import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader

from utils import DEVICE, MODEL_NAME
from utils import get_train_transforms, get_val_transforms, load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the training data (default: data)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: 0.0001)')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet50', 'repvgg_a0', 'repvgg_a1', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b2', 'repvgg_b3',
                                 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv3_large_100', 'mobilenetv3_small_100'],
                        help='Model architecture to use (default: xception)')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# training data loaders
transform_train = get_train_transforms()
train_dataset = datasets.ImageFolder(os.path.join(
    args.data_dir, "train"), transform=transform_train)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True)


# validation data loaders
transform_val = get_val_transforms()
val_dataset = datasets.ImageFolder(os.path.join(
    args.data_dir, "val"), transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


# model
model = load_model(model_name=args.model, num_classes=2, is_train=True)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True)


# Training Loop with best and last model saving, early stopping, and plotting
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_save_name, patience=7):
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # Lists to store accuracy and loss for plotting
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    print(f"Starting training for {epochs} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Early stopping patience: {patience} epochs")
    print("-" * 60)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # Store metrics for plotting
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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
                MODELS_DIR, f"{MODEL_NAME}_best_{epoch+1}.pth")
            torch.save(best_model_state, best_model_path)
            print(
                f"Best model saved as {best_model_path} with accuracy: {best_val_accuracy:.4f}")
        elif val_accuracy == best_val_accuracy:
            if train_accuracy > max(train_accuracies[:-1], default=0):
                best_model_state = model.state_dict().copy()
                print(
                    f"✓ Validation accuracy tied, but improved training accuracy: {train_accuracy:.4f}")
                epochs_without_improvement = 0
                best_model_path = os.path.join(
                    MODELS_DIR, f"{MODEL_NAME}_best_{epoch+1}.pth")
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

    return model, best_model_state, best_val_accuracy, train_accuracies, val_accuracies, train_losses, val_losses


# get date with format yyyymmdd
date_str = datetime.now().strftime("%Y%m%d")
model_save_name = f"{args.model}_{date_str}"

# Train the model
trained_model, best_model_state, best_accuracy, train_accs, val_accs, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, PATIENCE
    model, train_loader, val_loader, criterion, optimizer, args.epochs, model_save_name
)

# get date with format yyyymmdd
date_str = datetime.now().strftime("%Y%m%d")
model_save_name = f"{MODEL_NAME}_{date_str}"

# Save the models in the models directory
last_model_path = os.path.join(MODELS_DIR, f"{model_save_name}_last.pth")
best_model_path = os.path.join(MODELS_DIR, f"{model_save_name}_best.pth")

torch.save(trained_model.state_dict(), last_model_path)
print(f"Last model saved as {last_model_path}")

torch.save(best_model_state, best_model_path)
print(
    f"Best model saved as {best_model_path} with accuracy: {best_accuracy:.4f}")

# Plot training progress


def plot_training_progress(train_accs, val_accs, train_losses, val_losses, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    epochs_range = range(1, len(train_accs) + 1)
    ax1.plot(epochs_range, train_accs, 'b-', label='Training Accuracy')
    ax1.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    ax2.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(MODELS_DIR, f"{model_name}_training_progress.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved as {plot_path}")
    plt.show()


# Generate and save training plots
plot_training_progress(train_accs, val_accs, train_losses,
                       val_losses, model_save_name)


# Enhanced logging to a file
log_path = os.path.join(MODELS_DIR, f"{model_save_name}_log.txt")
with open(log_path, "w") as f:
    f.write(f"Training Log - {model_save_name}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Epochs Planned: {EPOCHS}\n")
    f.write(f"Total Epochs Completed: {len(train_accs)}\n")
    f.write(f"Early Stopping Patience: {PATIENCE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write("\n")
    f.write("Dataset Information:\n")
    f.write(f"Training samples: {len(train_loader.dataset)}\n")
    f.write(f"Validation samples: {len(val_loader.dataset)}\n")
    f.write(f"Classes: {train_loader.dataset.classes}\n")
    f.write("\n")
    f.write("Results:\n")
    f.write(f"Best validation accuracy: {best_accuracy:.4f}\n")
    f.write(f"Final training accuracy: {train_accs[-1]:.4f}\n")
    f.write(f"Final validation accuracy: {val_accs[-1]:.4f}\n")
    f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
    f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")
    f.write("\n")
    f.write("Model Files:\n")
    f.write(f"Best model: {best_model_path}\n")
    f.write(f"Last model: {last_model_path}\n")
    f.write(
        f"Training plot: {os.path.join(MODELS_DIR, f'{model_save_name}_training_progress.png')}\n")
    f.write("\n")
    f.write("Detailed Metrics:\n")
    f.write(f"train_accuracies: {train_accs}\n")
    f.write(f"val_accuracies: {val_accs}\n")
    f.write(f"train_losses: {train_losses}\n")
    f.write(f"val_losses: {val_losses}\n")

print(f"Training log saved to {log_path}")

# Print final summary
print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Best validation accuracy: {best_accuracy:.4f}")
print(f"Models saved in: {MODELS_DIR}/")
print(f"  - Best model: {model_save_name}_best.pth")
print(f"  - Last model: {model_save_name}_last.pth")
print(f"Training log: {log_path}")
print("=" * 60)
