import os
import cv2
import torch
import shutil
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils import DEVICE, CLASS_NAMES
from utils import load_model, get_val_transforms, model_predict


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a deep learning model')
    parser.add_argument('--data-dir', type=str, default="data/val",
                        help='Directory containing the validation data (default: data/val)')
    parser.add_argument('--model-path', type=str, default="./models/xception_20251003_best.pth",
                        help='Path to the trained model (default: ./models/xception_20251003_best.pth)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet50',
                                 'repvgg_a0', 'mobilenetv2_110d'],
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


def load_and_preprocess_image(image_path):
    """
    Load image and preprocess it for model prediction
    Handles color conversion from BGR to RGB if necessary
    """
    # Method 1: Load with PIL (automatically handles RGB)
    image = Image.open(image_path)

    # Convert to RGB if image is in different mode (RGBA, L, etc.)
    image = image.convert('RGB')

    return image


def predict_single_image(model, image_path, return_probs=False):
    """
    Predict on a single image with proper preprocessing
    """
    # Load and preprocess image
    image = load_and_preprocess_image(image_path)
    if image is None:
        return None

    # Make prediction using the existing model_predict function
    predicted_class, confidence, probabilities = model_predict(model, image)

    result = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'predicted_label': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities
    }

    if return_probs:
        result['all_probabilities'] = probabilities

    return result


def predict_folder_images(model, folder_path, class_name=None):
    """
    Predict on all images in a folder
    If class_name is provided, it's used as ground truth for accuracy calculation
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp',
                        '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF']

    folder_path = Path(folder_path)
    results = []

    for ext in image_extensions:
        for image_path in folder_path.glob(f'*{ext}'):
            result = predict_single_image(
                model, str(image_path), return_probs=True)
            if result is not None:
                # Add ground truth if provided
                if class_name is not None:
                    if class_name in CLASS_NAMES:
                        result['true_class'] = class_name
                        result['true_label'] = CLASS_NAMES.index(class_name)
                    else:
                        print(
                            f"Warning: Unknown class name '{class_name}'. Available classes: {CLASS_NAMES}")

                results.append(result)

    return results


def calculate_accuracy_from_results(results):
    """
    Calculate accuracy metrics from prediction results
    """
    if not results or 'true_label' not in results[0]:
        print("No ground truth labels found in results. Cannot calculate accuracy.")
        return None

    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted_class'] for r in results]

    # Calculate overall accuracy
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    accuracy = correct / len(y_true)

    # Calculate per-class accuracy
    class_accuracies = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_correct = sum(1 for i in range(len(y_true))
                            if y_true[i] == class_idx and y_pred[i] == class_idx)
        class_total = sum(1 for label in y_true if label == class_idx)
        class_accuracies[class_name] = class_correct / \
            class_total if class_total > 0 else 0

    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'y_true': y_true,
        'y_pred': y_pred,
        'total_samples': len(results),
        'correct_predictions': correct
    }


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


def evaluate_with_proper_preprocessing(model, data_path):
    """
    Evaluate model with proper image loading and preprocessing
    This ensures images are loaded and preprocessed the same way as in training
    """
    results = []

    # Process each class folder
    for class_name in CLASS_NAMES:
        class_folder = Path(data_path) / class_name
        if class_folder.exists():
            print(f"Processing {class_name} images from {class_folder}")
            class_results = predict_folder_images(
                model, class_folder, class_name)
            results.extend(class_results)
            print(f"Found {len(class_results)} {class_name} images")
        else:
            print(f"Warning: Class folder {class_folder} does not exist")

    if not results:
        print(f"No images found in {data_path}")
        return None, None, []

    # Calculate accuracy
    accuracy_metrics = calculate_accuracy_from_results(results)

    if accuracy_metrics:
        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS WITH PROPER PREPROCESSING")
        print(f"{'='*60}")
        print(f"Total images processed: {accuracy_metrics['total_samples']}")
        print(
            f"Overall accuracy: {accuracy_metrics['overall_accuracy']:.4f} ({accuracy_metrics['overall_accuracy']*100:.2f}%)")
        print(
            f"Correct predictions: {accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_samples']}")

        print("\nPer-class accuracy:")
        for class_name, acc in accuracy_metrics['class_accuracies'].items():
            class_count = sum(1 for r in results if r.get(
                'true_class') == class_name)
            print(
                f"  {class_name}: {acc:.4f} ({acc*100:.2f}%) - {class_count} images")

        # Get misclassified images
        error_files = []
        for result in results:
            if result['predicted_class'] != result['true_label']:
                error_files.append(result['image_path'])

        return accuracy_metrics['y_true'], accuracy_metrics['y_pred'], error_files

    return None, None, []


# Method 1: Evaluate using DataLoader (original method)
print("="*60)
print("METHOD 1: Using DataLoader (original)")
print("="*60)
labels, preds, error_files = evaluate_model(model, val_loader, val_dataset)
cm = confusion_matrix(labels, preds)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(labels, preds, target_names=val_dataset.classes))

# Method 2: Evaluate with proper preprocessing (recommended)
print(f"\n{'='*60}")
print("METHOD 2: Using proper image loading and preprocessing")
print(f"{'='*60}")
labels_v2, preds_v2, error_files_v2 = evaluate_with_proper_preprocessing(
    model, DATA_SET_PATH)

if labels_v2 is not None and preds_v2 is not None:
    cm_v2 = confusion_matrix(labels_v2, preds_v2)
    print("\nConfusion Matrix (Method 2):")
    print(cm_v2)
    print("Classification Report (Method 2):")
    print(classification_report(labels_v2, preds_v2, target_names=CLASS_NAMES))

    # Compare methods
    if len(labels) == len(labels_v2):
        print(f"\nComparison between methods:")
        print(f"Method 1 accuracy: {np.mean(labels == preds):.4f}")
        print(
            f"Method 2 accuracy: {np.mean(np.array(labels_v2) == np.array(preds_v2)):.4f}")

        # Check if predictions differ
        if not np.array_equal(preds, preds_v2):
            diff_count = np.sum(np.array(preds) != np.array(preds_v2))
            print(
                f"Predictions differ on {diff_count} images out of {len(labels)}")
        else:
            print("Both methods produced identical predictions")

    # Use the better method's results for further analysis
    labels, preds, error_files = labels_v2, preds_v2, error_files_v2
    cm = cm_v2

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

# Example: Test individual images with proper preprocessing
print(f"\n{'='*60}")
print("INDIVIDUAL IMAGE PREDICTION EXAMPLES")
print(f"{'='*60}")

# Test a few individual images if they exist
test_images = []
for class_name in CLASS_NAMES:
    class_folder = Path(DATA_SET_PATH) / class_name
    if class_folder.exists():
        # Get first image from each class
        for ext in ['.png', '.jpg', '.jpeg']:
            images = list(class_folder.glob(f'*{ext}'))
            if images:
                test_images.append((str(images[0]), class_name))
                break

for image_path, true_class in test_images:
    result = predict_single_image(model, image_path, return_probs=True)
    if result:
        print(f"\nImage: {Path(image_path).name}")
        print(f"True class: {true_class}")
        print(
            f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
        print(
            f"Probabilities: Benign={result['probabilities'][0]:.3f}, Cancer={result['probabilities'][1]:.3f}")

        # Check if prediction is correct
        is_correct = result['predicted_label'] == true_class
        print(f"Prediction: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

print(f"\n{'='*60}")
print("TIPS FOR USING THIS CODE:")
print(f"{'='*60}")
print("1. The new functions handle proper image loading and color conversion")
print("2. Use predict_single_image() for individual image predictions")
print("3. Use predict_folder_images() for batch predictions on a folder")
print("4. Use evaluate_with_proper_preprocessing() for accurate evaluation")
print("5. Images are automatically converted from BGR to RGB when needed")
print("6. The code handles various image formats: PNG, JPG, JPEG, BMP, TIFF")
