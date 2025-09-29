import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ensemble_model import create_ensemble_model, get_ensemble_transforms
from utils import DEVICE, CLASS_NAMES


class EnsemblePredictor:
    """
    Prediction class for ensemble models
    """

    def __init__(self, model_path, ensemble_method='average'):
        self.ensemble_method = ensemble_method
        self.device = DEVICE

        # Load model
        self.model = create_ensemble_model(ensemble_method=ensemble_method)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Get transforms
        _, self.transform = get_ensemble_transforms()

        print(f"Loaded {ensemble_method} ensemble model from {model_path}")

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

    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with probabilities
        """
        result = self.predict_single_image(image_path)

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


def predict_on_test_folder(model_path, test_folder, ensemble_method='average'):
    """
    Predict on all images in a test folder
    """
    import os
    from pathlib import Path

    predictor = EnsemblePredictor(model_path, ensemble_method)

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(test_folder).rglob(f'*{ext}'))
        image_paths.extend(Path(test_folder).rglob(f'*{ext.upper()}'))

    image_paths = [str(p) for p in image_paths]

    if not image_paths:
        print(f"No images found in {test_folder}")
        return []

    print(f"Found {len(image_paths)} images")

    # Predict on all images
    results = predictor.predict_batch(image_paths)

    # Summary statistics
    predictions = [r['predicted_class'] for r in results]
    benign_count = predictions.count(0)
    cancer_count = predictions.count(1)

    print(f"\nPrediction Summary:")
    print(
        f"Benign: {benign_count} images ({benign_count/len(results)*100:.1f}%)")
    print(
        f"Cancer: {cancer_count} images ({cancer_count/len(results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    import os

    # Example usage
    # Update with your model path
    model_path = "models/ensemble_average_20250929_best.pth"
    ensemble_method = "average"

    # Single image prediction
    image_path = "/home/cybercore/Workspaces/phitv/classify/data/val/Cancer/04.06.2025_1 (1).png"

    # if os.path.exists(image_path):
    #     predictor = EnsemblePredictor(model_path, ensemble_method)
    #     result = predictor.visualize_prediction(image_path)

    #     # Compare individual models
    #     comparison = predictor.compare_individual_models(image_path)

    # Predict on test folder
    test_folder = "data/val"
    results = predict_on_test_folder(model_path, test_folder, ensemble_method)
    print(results)
