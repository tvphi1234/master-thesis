import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


# validation data loaders
transform_val = get_val_transforms()
val_dataset = datasets.ImageFolder(os.path.join(
    args.data_dir, "val"), transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


# model
model = load_model(model_name=args.model, num_classes=2, is_train=True)


# Loss and Optimizer
# Use BCEWithLogitsLoss for binary classification with sigmoid
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


# Training Loop with best and last model saving and plotting
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_save_name):
    best_val_accuracy = 0.0
    best_model_state = None

    # Lists to store accuracy and loss for plotting
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"{model_save_name}_best_{epoch+1}.pth")
            print(
                f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
            # save the new best model
        elif val_accuracy == best_val_accuracy and train_accuracy > max(train_accuracies[:-1]):
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"{model_save_name}_best_{epoch+1}.pth")
            print(
                f"New best model saved with validation accuracy: {best_val_accuracy:.4f} and improved training accuracy: {train_accuracy:.4f}")

    return model, best_model_state, best_val_accuracy, train_accuracies, val_accuracies, train_losses, val_losses


# get date with format yyyymmdd
date_str = datetime.now().strftime("%Y%m%d")
model_save_name = f"{args.model}_{date_str}"

# Train the model
trained_model, best_model_state, best_accuracy, train_accs, val_accs, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, args.epochs, model_save_name
)

# Save the models
torch.save(trained_model.state_dict(), f"{model_save_name}_last.pth")
print(f"Last model saved as {model_save_name}_last.pth")

torch.save(best_model_state, f"{model_save_name}_best.pth")
print(
    f"Best model saved as {model_save_name}_best.pth with accuracy: {best_accuracy:.4f}")


# logging to a file
with open(f"{model_save_name}_log.txt", "w") as f:
    f.write(f"Training completed for {args.epochs} epochs\n")
    f.write(f"Last model: {model_save_name}_last.pth\n")
    f.write(
        f"Best model: {model_save_name}_best.pth with accuracy: {best_accuracy:.4f}\n")
    f.write(f"train_accuracies: {train_accs}\n")
    f.write(f"val_accuracies: {val_accs}\n")
    f.write(f"train_losses: {train_losses}\n")
    f.write(f"val_losses: {val_losses}\n")
print(f"Training log saved to {model_save_name}_log.txt")
