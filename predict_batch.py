import os
import math

from PIL import Image
from pathlib import Path

from utils import DEVICE, CLASS_NAMES
from utils import load_model, model_predict

# Parameters
BATCH_SIZE = 1
DATA_SET_PATH = "dataset/x10/04.06.2025/Benign"
MODEL_PATH = "models_x40_public/xception_20251119_best.pth"

# Load model
model = load_model(num_classes=2,
                   model_path=MODEL_PATH,
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


def predict_single_image(model, image, image_path=None, return_probs=False):
    """
    Predict on a single image with proper preprocessing
    """

    if not image:
        image = load_and_preprocess_image(image_path)

    # Ensure image is loaded
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


def split_image_to_patches(image, num_patches):
    """
    Split the input image into the specified number of patches
    Assumes num_patches is a perfect square (e.g., 4, 9, 16)
    """

    # Calculate number of rows and columns
    patches_per_side = int(math.sqrt(num_patches))
    img_width, img_height = image.size

    patch_width = img_width // patches_per_side
    patch_height = img_height // patches_per_side

    patches = []
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            left = j * patch_width
            upper = i * patch_height
            right = (j + 1) * patch_width if (j + 1) * \
                patch_width <= img_width else img_width
            lower = (i + 1) * patch_height if (i + 1) * \
                patch_height <= img_height else img_height

            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
    return patches


if __name__ == "__main__":

    for root, dirs, files in os.walk(DATA_SET_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)

                image = load_and_preprocess_image(image_path)

                # split image to 8 patches
                patches = split_image_to_patches(image, 25)

                cancer = []
                for patch in patches:
                    result = predict_single_image(
                        model, patch, image_path=image_path)

                    # tune label with threshold 0.9999
                    if result['confidence'] < 0.9999:
                        result['predicted_label'] = 'Benign'

                    cancer.append(result['predicted_label'])

                    # save patch image
                    patch_filename = f"{os.path.splitext(file)[0]}_patch_{len(cancer)}_{result['predicted_label']}.png"
                    patch.save(patch_filename)

                # print(f"Image: {image_path}, Cancer predictions: {cancer}")

                if cancer.count('Cancer') >= 4:
                    final_label = 'Cancer'
                else:
                    final_label = 'Benign'

                print(
                    f"Final prediction for image {image_path}: {final_label}")
