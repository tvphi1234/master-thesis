import torch
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from models import CustomModel
from utils import DEVICE, CLASS_NAMES, STAGE_NAMES
from dataloader import CustomImageDataset, get_dataloader, get_val_transforms


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a deep learning model')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--data-dir', type=str, default="data",
                        help='Directory containing the validation data (default: data/val)')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet50',
                                 'repvgg_a0', 'mobilenetv2_110d'],
                        help='Model architecture to use (default: xception)')
    parser.add_argument('--model-path', type=str, default="./models/xception_20251003_best.pth",
                        help='Path to the trained model (default: ./models/xception_20251003_best.pth)')
    parser.add_argument('--task-type', type=str, default='multi_task',
                        choices=['cancer', 'stage', 'multi_task'],
                        help='Task type for the model (default: cancer)')
    return parser.parse_args()


# Parse command line arguments
args = get_args()


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


if __name__ == "__main__":

    # model
    model = CustomModel(model_name=args.model,
                        task_type=args.task_type).to(DEVICE)
    model.load_pretrained_weights(args.model_path)

    test_transform = get_val_transforms()
    test_loader = get_dataloader(
        data_dir=args.data_dir,
        annotation_files=["x10_test_annotations.csv", "x10_warwick_test_annotations.csv"],
        data_transform=test_transform,
        is_shuffle=False,
        batch_size=1,
        task_type=args.task_type
    )

    model.eval()
    cancer_true, cancer_pred_list = [], []
    level_true, level_pred_list = [], []

    with torch.no_grad():
        for inputs, labels, levels in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            levels = levels.to(DEVICE)

            if args.task_type == 'multi_task':

                cancer_probs, level_probs = model.predict(inputs)
                cancer_preds = torch.argmax(cancer_probs, dim=1)
                level_preds = torch.argmax(level_probs, dim=1)

                # collect cancer labels/preds
                cancer_true.extend(labels.cpu().tolist())
                cancer_pred_list.extend(cancer_preds.cpu().tolist())

                # only evaluate level where label is present (>0)
                valid_mask = levels >= 0
                if valid_mask.any():
                    # adjust level index: original labels assumed 1..K -> 0..K-1
                    level_targets = levels[valid_mask].cpu()
                    level_pred_valid = level_preds[valid_mask].cpu()
                    level_true.extend(level_targets.tolist())
                    level_pred_list.extend(level_pred_valid.tolist())
            else:
                probs = model.predict(inputs)
                preds = torch.argmax(probs, dim=1)

                if args.task_type == 'stage':
                    # only evaluate level where label is present (>0)
                    level_true.extend(levels.cpu().tolist())  # adjust index
                    level_pred_list.extend(preds.cpu().tolist())

                else:  # cancer task
                    cancer_true.extend(labels.cpu().tolist())
                    cancer_pred_list.extend(preds.cpu().tolist())

    if cancer_true:
        # Confusion matrix for cancer classification
        cancer_cm = confusion_matrix(
            cancer_true, cancer_pred_list, labels=list(range(len(CLASS_NAMES))))
        print("Cancer confusion matrix (rows=true, cols=pred):")
        print(cancer_cm)
        print(classification_report(cancer_true,
                                    cancer_pred_list, target_names=CLASS_NAMES))

    # Confusion matrix for level prediction (only for valid labels)
    if level_true:
        level_cm = confusion_matrix(level_true, level_pred_list)
        print("Level confusion matrix (rows=true, cols=pred):")
        print(level_cm)
        level_target_names = STAGE_NAMES if 'STAGE_NAMES' in globals() else None
        print(classification_report(level_true,
              level_pred_list, target_names=level_target_names))
    else:
        print("No valid level labels (>0) found; skipped level confusion matrix.")
