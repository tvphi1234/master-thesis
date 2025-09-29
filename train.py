import os
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torchvision import datasets
from torch.utils.data import DataLoader

from utils import DEVICE, MODEL_NAME
from utils import get_train_transforms, get_val_transforms, load_model


# Parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = "data"


# training data loaders
transform_train = get_train_transforms()
train_dataset = datasets.ImageFolder(os.path.join(
    DATA_DIR, "train"), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# validation data loaders
transform_val = get_val_transforms()
val_dataset = datasets.ImageFolder(os.path.join(
    DATA_DIR, "val"), transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# model
model = load_model(model_name=MODEL_NAME, num_classes=2, is_train=True)


# Loss and Optimizer
# Use BCEWithLogitsLoss for binary classification with sigmoid
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training Loop with best and last model saving and plotting
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
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
            print(
                f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    return model, best_model_state, best_val_accuracy, train_accuracies, val_accuracies, train_losses, val_losses


# Train the model
trained_model, best_model_state, best_accuracy, train_accs, val_accs, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS
)

# get date with format yyyymmdd
date_str = datetime.now().strftime("%Y%m%d")
MODEL_NAME = f"{MODEL_NAME}_{date_str}"

# Save the models
torch.save(trained_model.state_dict(), f"{MODEL_NAME}_last.pth")
print(f"Last model saved as {MODEL_NAME}_last.pth")

torch.save(best_model_state, f"{MODEL_NAME}_best.pth")
print(
    f"Best model saved as {MODEL_NAME}_best.pth with accuracy: {best_accuracy:.4f}")


# logging to a file
with open(f"{MODEL_NAME}_log.txt", "w") as f:
    f.write(f"Training completed for {EPOCHS} epochs\n")
    f.write(f"Last model: {MODEL_NAME}_last.pth\n")
    f.write(
        f"Best model: {MODEL_NAME}_best.pth with accuracy: {best_accuracy:.4f}\n")
    f.write(f"train_accuracies: {train_accs}\n")
    f.write(f"val_accuracies: {val_accs}\n")
    f.write(f"train_losses: {train_losses}\n")
    f.write(f"val_losses: {val_losses}\n")
print("Training log saved to training_log.txt")
