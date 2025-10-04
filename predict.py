import os
import torch
import shutil
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils import DEVICE
from utils import load_model, get_val_transforms

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a deep learning model')
    parser.add_argument('--data-dir', type=str, default="data/val",
                        help='Directory containing the validation data (default: data/val)')
    parser.add_argument('--model-path', type=str, default="./models/xception_20251003_best.pth",
                        help='Path to the trained model (default: ./models/xception_20251003_best.pth)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet50', 'repvgg_a0'],
                        help='Model architecture to use (default: xception)')
    return parser.parse_args()

# Parse command line arguments
args = get_args()

# Data Preparation
val_transform = get_val_transforms()

# Load validation dataset
val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Load model
model = load_model(model_name=args.model,
                   num_classes=2,
                   model_path=args.model_path,
                   is_train=False)


# Evaluation function with error image collection
def evaluate_model(model, data_loader, dataset):
    all_preds = []
    all_labels = []
    error_files = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            # Get indices of misclassified images in this batch
            mis_idx = (preds.cpu() != labels).nonzero(as_tuple=True)[0]
            # Calculate global indices for this batch
            start_idx = batch_idx * data_loader.batch_size
            for i in mis_idx:
                img_idx = start_idx + i.item()
                # Get the image path from the dataset
                img_path, _ = dataset.samples[img_idx]
                error_files.append(img_path)
    return np.array(all_labels), np.array(all_preds), error_files


# Get prediction probabilities instead of just predictions
def get_probabilities(model, data_loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            # Probability of positive class
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


# Evaluate and print confusion matrix and classification report
labels, preds, error_files = evaluate_model(model, val_loader, val_dataset)
cm = confusion_matrix(labels, preds)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(labels, preds, target_names=val_dataset.classes))

# Print misclassified image paths
print("Misclassified image files:")
for f in error_files:
    # copy the file to here
    print(f)
    shutil.copy2(f, os.path.basename(f))


# Đánh giá độ nhạy, độ đặc hiệu, F1-score
if cm.shape == (2, 2):  # Chỉ áp dụng cho phân loại nhị phân
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Độ nhạy (Recall)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Độ đặc hiệu
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP +
                                         FP + FN) > 0 else 0  # F1-score

    print(f"Độ nhạy (Recall): {sensitivity:.4f}")
    print(f"Độ đặc hiệu (Specificity): {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("Chỉ số độ nhạy, độ đặc hiệu, F1-score chỉ áp dụng cho phân loại nhị phân.")

# Save confusion matrix as an image
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# Calculate ROC and AUC
labels, probs = get_probabilities(model, val_loader)
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

print(thresholds)
print(f"AUC Score: {roc_auc:.3f}")
