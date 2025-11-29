import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image
from ensemble_model import create_ensemble_model, get_ensemble_transforms
from utils import DEVICE, CLASS_NAMES

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True


class EnsemblePredictor:
    """
    Prediction class for ensemble models and single models
    """

    def __init__(self, model_path, ensemble_method='average'):
        self.ensemble_method = ensemble_method
        self.device = DEVICE

        if ensemble_method == 'single':
            # Load single model (ResNet50 or Xception) using timm
            from timm import create_model

            if 'resnet' in model_path.lower():
                self.model = create_model(
                    'resnet50', pretrained=False, num_classes=len(CLASS_NAMES))
                model_type = "ResNet50"
            elif 'xception' in model_path.lower():
                self.model = create_model(
                    'xception', pretrained=False, num_classes=len(CLASS_NAMES))
                model_type = "Xception"
            else:
                # Default to ResNet50 if model type cannot be determined
                self.model = create_model(
                    'resnet50', pretrained=False, num_classes=len(CLASS_NAMES))
                model_type = "ResNet50 (default)"

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {model_type} single model from {model_path}")
        else:
            # Load ensemble model
            self.model = create_ensemble_model(ensemble_method=ensemble_method)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {ensemble_method} ensemble model from {model_path}")

        self.model.eval()

        # Get transforms
        _, self.transform = get_ensemble_transforms()

    def predict_single_image(self, image_path):
        """
        Predict on a single image
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()

        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get ensemble prediction
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            # Get individual model predictions if available
            individual_preds = None
            if hasattr(self.model, 'get_individual_predictions'):
                try:
                    individual_preds = self.model.get_individual_predictions(
                        input_tensor)
                except:
                    individual_preds = None

        result = {
            'predicted_class': predicted_class.item(),
            'predicted_label': CLASS_NAMES[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy(),
            'individual_predictions': individual_preds,
            'original_image': original_image
        }

        return result

    def copy_misclassified_image(self, image_path, true_label, predicted_class, predicted_label):
        """
        Copy a misclassified image to the cache folder
        """
        import shutil
        from pathlib import Path

        if true_label != predicted_class:
            # Create cache directory if it doesn't exist
            cache_dir = Path("cache")
            wrong_predict_dir = cache_dir / "wrong_predict"
            wrong_predict_dir.mkdir(parents=True, exist_ok=True)

            # Copy the image
            src_path = Path(image_path)
            dst_path = wrong_predict_dir / src_path.name

            try:
                shutil.copy2(src_path, dst_path)
                print(
                    f"Copied misclassified image: {src_path.name} (True: {CLASS_NAMES[true_label]}, Predicted: {predicted_label})")
                return dst_path
            except Exception as e:
                print(f"Error copying {src_path.name}: {e}")
                return None
        return None

    def predict_batch(self, image_paths):
        """
        Predict on multiple images
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return results

    def visualize_prediction(self, image_path, save_path=None, true_label=None, copy_if_wrong=True):
        """
        Visualize prediction with probabilities
        """
        result = self.predict_single_image(image_path)

        # Copy misclassified image if true label is provided
        if true_label is not None and copy_if_wrong:
            self.copy_misclassified_image(image_path, true_label,
                                          result['predicted_class'],
                                          result['predicted_label'])

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Show original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title(f"Input Image\n{image_path.split('/')[-1]}")
        axes[0].axis('off')

        # Show prediction probabilities
        bars = axes[1].bar(CLASS_NAMES, result['probabilities'])
        axes[1].set_title(f"Ensemble Prediction ({self.ensemble_method.title()})\n"
                          f"Predicted: {result['predicted_label']} "
                          f"(Confidence: {result['confidence']:.3f})")
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1)

        # Color bars based on prediction
        for i, bar in enumerate(bars):
            if i == result['predicted_class']:
                bar.set_color('green')
            else:
                bar.set_color('red')

        # Add probability values on bars
        for i, prob in enumerate(result['probabilities']):
            axes[1].text(i, prob + 0.01, f'{prob:.3f}',
                         ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return result

    def compare_individual_models(self, image_path):
        """
        Compare predictions from individual models in the ensemble
        """
        if self.ensemble_method == 'feature_fusion':
            print("Individual model comparison not available for feature_fusion method")
            return None

        result = self.predict_single_image(image_path)

        if result['individual_predictions'] is None:
            print("Individual predictions not available")
            return result

        resnet_probs, xception_probs = result['individual_predictions']

        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Show original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title(f"Input Image\n{image_path.split('/')[-1]}")
        axes[0].axis('off')

        # ResNet50 predictions
        bars1 = axes[1].bar(CLASS_NAMES, resnet_probs[0])
        resnet_pred = np.argmax(resnet_probs[0])
        axes[1].set_title(f"ResNet50 Prediction\n"
                          f"Predicted: {CLASS_NAMES[resnet_pred]} "
                          f"({resnet_probs[0][resnet_pred]:.3f})")
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1)

        for i, bar in enumerate(bars1):
            if i == resnet_pred:
                bar.set_color('blue')
            else:
                bar.set_color('lightblue')

        # Xception predictions
        bars2 = axes[2].bar(CLASS_NAMES, xception_probs[0])
        xception_pred = np.argmax(xception_probs[0])
        axes[2].set_title(f"Xception Prediction\n"
                          f"Predicted: {CLASS_NAMES[xception_pred]} "
                          f"({xception_probs[0][xception_pred]:.3f})")
        axes[2].set_ylabel('Probability')
        axes[2].set_ylim(0, 1)

        for i, bar in enumerate(bars2):
            if i == xception_pred:
                bar.set_color('orange')
            else:
                bar.set_color('moccasin')

        plt.tight_layout()
        plt.show()

        # Print comparison
        print(f"\nModel Comparison for {image_path.split('/')[-1]}:")
        print(
            f"ResNet50:  {CLASS_NAMES[resnet_pred]} (confidence: {resnet_probs[0][resnet_pred]:.3f})")
        print(
            f"Xception:  {CLASS_NAMES[xception_pred]} (confidence: {xception_probs[0][xception_pred]:.3f})")
        print(
            f"Ensemble:  {result['predicted_label']} (confidence: {result['confidence']:.3f})")

        return {
            'ensemble': result,
            'resnet50': {'prediction': resnet_pred, 'probabilities': resnet_probs[0]},
            'xception': {'prediction': xception_pred, 'probabilities': xception_probs[0]}
        }


def predict_on_test_folder(model_path, test_folder, ensemble_method='average', copy_misclassified=True):
    """
    Predict on all images in a test folder and calculate accuracy
    """
    import os
    import shutil
    from pathlib import Path

    predictor = EnsemblePredictor(model_path, ensemble_method)

    # Get all image files with their true labels
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_data = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = Path(test_folder) / class_name
        if class_folder.exists():
            for ext in image_extensions:
                for img_path in class_folder.glob(f'*{ext}'):
                    image_data.append({
                        'path': str(img_path),
                        'true_label': class_idx,
                        'true_class': class_name
                    })
                for img_path in class_folder.glob(f'*{ext.upper()}'):
                    image_data.append({
                        'path': str(img_path),
                        'true_label': class_idx,
                        'true_class': class_name
                    })

    if not image_data:
        print(f"No images found in {test_folder}")
        return []

    print(f"Found {len(image_data)} images")
    print(f"Class distribution:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        count = sum(
            1 for item in image_data if item['true_label'] == class_idx)
        print(f"  {class_name}: {count} images")

    # Predict on all images
    image_paths = [item['path'] for item in image_data]
    results = predictor.predict_batch(image_paths)

    # Add ground truth labels to results
    for i, result in enumerate(results):
        result['true_label'] = image_data[i]['true_label']
        result['true_class'] = image_data[i]['true_class']

    # Copy misclassified images to cache folder
    if copy_misclassified:
        # Create cache directories if they don't exist
        cache_dir = Path("cache")
        wrong_predict_dir = cache_dir / "wrong_predict"
        true_predict_dir = cache_dir / "true_predict"
        wrong_predict_dir.mkdir(parents=True, exist_ok=True)
        true_predict_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            if result['true_label'] != result['predicted_class']:
                # Copy misclassified image
                src_path = Path(result['image_path'])
                dst_path = wrong_predict_dir / src_path.name
                try:
                    shutil.copy2(src_path, dst_path)
                    print(
                        f"Copied misclassified image: {src_path.name} (True: {result['true_class']}, Predicted: {result['predicted_label']})")
                except Exception as e:
                    print(f"Error copying {src_path.name}: {e}")
            # else:
            #     # Optionally copy correctly classified images
            #     src_path = Path(result['image_path'])
            #     dst_path = true_predict_dir / src_path.name
            #     try:
            #         shutil.copy2(src_path, dst_path)
            #     except Exception as e:
            #         print(
            #             f"Error copying correctly classified {src_path.name}: {e}")

    # Calculate accuracy metrics
    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted_class'] for r in results]

    accuracy = accuracy_score(y_true, y_pred)

    # Calculate per-class accuracy
    benign_correct = sum(1 for i in range(len(y_true))
                         if y_true[i] == 0 and y_pred[i] == 0)
    benign_total = sum(1 for label in y_true if label == 0)
    cancer_correct = sum(1 for i in range(len(y_true))
                         if y_true[i] == 1 and y_pred[i] == 1)
    cancer_total = sum(1 for label in y_true if label == 1)

    benign_accuracy = benign_correct / benign_total if benign_total > 0 else 0
    cancer_accuracy = cancer_correct / cancer_total if cancer_total > 0 else 0

    # Print detailed results
    print(f"\n{'='*50}")
    print(f"ACCURACY RESULTS - {ensemble_method.upper()} ENSEMBLE")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(
        f"Benign Accuracy: {benign_accuracy:.4f} ({benign_accuracy*100:.2f}%) - {benign_correct}/{benign_total}")
    print(
        f"Cancer Accuracy: {cancer_accuracy:.4f} ({cancer_accuracy*100:.2f}%) - {cancer_correct}/{cancer_total}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Benign  Cancer")
    print(f"Actual Benign    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Cancer    {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=CLASS_NAMES, digits=4))

    # Summary statistics
    predictions = [r['predicted_class'] for r in results]
    benign_pred_count = predictions.count(0)
    cancer_pred_count = predictions.count(1)

    print(f"\nPrediction Summary:")
    print(
        f"Predicted Benign: {benign_pred_count} images ({benign_pred_count/len(results)*100:.1f}%)")
    print(
        f"Predicted Cancer: {cancer_pred_count} images ({cancer_pred_count/len(results)*100:.1f}%)")

    return results, {
        'accuracy': accuracy,
        'benign_accuracy': benign_accuracy,
        'cancer_accuracy': cancer_accuracy,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }


def compare_model_accuracies(test_folder, model_configs, copy_misclassified=True):
    """
    Compare accuracies of multiple models

    Args:
        test_folder: Path to test data
        model_configs: List of dictionaries with 'path', 'method', and 'name' keys
        copy_misclassified: Whether to copy misclassified images to cache folder
    """
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON RESULTS")
    print(f"{'='*60}")

    results_summary = []

    for config in model_configs:
        print(f"\nTesting {config['name']}...")
        _, metrics = predict_on_test_folder(
            config['path'],
            test_folder,
            config['method'],
            copy_misclassified
        )

        results_summary.append({
            'name': config['name'],
            'accuracy': metrics['accuracy'],
            'benign_accuracy': metrics['benign_accuracy'],
            'cancer_accuracy': metrics['cancer_accuracy'],
            'confusion_matrix': metrics['confusion_matrix']
        })

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Overall':<10} {'Benign':<10} {'Cancer':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for result in results_summary:
        print(f"{result['name']:<25} "
              f"{result['accuracy']:.4f}    "
              f"{result['benign_accuracy']:.4f}    "
              f"{result['cancer_accuracy']:.4f}")

    # Find best model
    best_overall = max(results_summary, key=lambda x: x['accuracy'])
    best_benign = max(results_summary, key=lambda x: x['benign_accuracy'])
    best_cancer = max(results_summary, key=lambda x: x['cancer_accuracy'])

    print(f"\n{'='*60}")
    print(f"BEST PERFORMING MODELS")
    print(f"{'='*60}")
    print(
        f"Best Overall Accuracy: {best_overall['name']} ({best_overall['accuracy']:.4f})")
    print(
        f"Best Benign Accuracy:  {best_benign['name']} ({best_benign['benign_accuracy']:.4f})")
    print(
        f"Best Cancer Accuracy:  {best_cancer['name']} ({best_cancer['cancer_accuracy']:.4f})")

    return results_summary


def analyze_misclassified_images():
    """
    Analyze the misclassified images in the cache folder
    """
    import os
    from pathlib import Path

    cache_dir = Path("cache")
    wrong_predict_dir = cache_dir / "wrong_predict"
    true_predict_dir = cache_dir / "true_predict"

    if not wrong_predict_dir.exists():
        print("No misclassified images found in cache/wrong_predict/")
        return

    wrong_images = list(wrong_predict_dir.glob("*.png")) + \
        list(wrong_predict_dir.glob("*.jpg"))
    true_images = list(true_predict_dir.glob(
        "*.png")) + list(true_predict_dir.glob("*.jpg")) if true_predict_dir.exists() else []

    print(f"\n{'='*50}")
    print(f"MISCLASSIFIED IMAGES ANALYSIS")
    print(f"{'='*50}")
    print(f"Total misclassified images: {len(wrong_images)}")
    print(f"Total correctly classified images: {len(true_images)}")

    if wrong_images:
        print(f"\nMisclassified images:")
        for img in sorted(wrong_images):
            print(f"  - {img.name}")

    return {
        'misclassified_count': len(wrong_images),
        'correct_count': len(true_images),
        'misclassified_files': [img.name for img in wrong_images],
        'correct_files': [img.name for img in true_images]
    }


if __name__ == "__main__":
    import os

    # Test folder
    test_folder = "data/val"

    # Model configurations for comparison
    model_configs = [
        # {
        #     'path': 'models/ensemble_average_20250929_best.pth',
        #     'method': 'average',
        #     'name': 'Ensemble Average'
        # },
        {
            'path': 'models/ensemble_feature_fusion_20250928_best.pth',
            'method': 'feature_fusion',
            'name': 'Ensemble Feature Fusion'
        }
    ]

    # Filter only existing models
    available_configs = []
    for config in model_configs:
        if os.path.exists(config['path']):
            available_configs.append(config)
        else:
            print(f"Model not found: {config['path']}")

    if available_configs:
        # Compare all available models
        comparison_results = compare_model_accuracies(
            test_folder, available_configs)
    else:
        print("No model files found. Please check the model paths.")

    # Example single model test (uncomment to use)
    # model_path = "models/ensemble_average_20250929_best.pth"
    # ensemble_method = "average"
    # results, metrics = predict_on_test_folder(model_path, test_folder, ensemble_method)
