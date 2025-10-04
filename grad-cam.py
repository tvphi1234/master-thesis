import os
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import load_model, visualize_grad_cam


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
    parser.add_argument('--output-dir', type=str, default="./grad_cam_results",
                        help='Directory to save Grad-CAM results (default: ./grad_cam_results)')
    return parser.parse_args()

# Parse command line arguments
args = get_args()

os.makedirs(args.output_dir, exist_ok=True)


model = load_model(model_name=args.model,
                   num_classes=2,
                   model_path=args.model_path,
                   is_train=False)

# Select appropriate layer based on model architecture
def get_target_layers(model, model_name):
    """
    Get the appropriate target layer for Grad-CAM based on model architecture
    """
    if model_name == 'xception':
        # For Xception, use the last activation layer
        return [model.act4]
    elif model_name == 'resnet50':
        # For ResNet50, use the last convolutional layer (layer4 - the last residual block)
        return [model.layer4[-1]]  # Last block of layer4
    elif model_name.startswith('repvgg'):
        # For RepVGG models, use the last convolutional stage
        return [model.stages[-1]]  # Last stage



def print_model_structure(model, max_depth=2):
    """
    Print model structure to help debug layer names
    """
    print("Model structure:")
    for name, module in model.named_modules():
        depth = name.count('.')
        if depth <= max_depth:
            indent = "  " * depth
            print(f"{indent}{name}: {type(module).__name__}")

# Uncomment the next line if you want to see the model structure
# print_model_structure(model)

target_layers = get_target_layers(model, args.model)

# Print information about the selected layer
print(f"Model: {args.model}")
print(f"Selected target layer: {target_layers[0]}")
print(f"Layer type: {type(target_layers[0])}")

# Get the layer name for debugging
layer_name = "unknown"
for name, module in model.named_modules():
    if module is target_layers[0]:
        layer_name = name
        break
print(f"Layer name: {layer_name}")

# Grad-CAM
grad_cam = GradCAM(model=model, target_layers=target_layers)

# 1 là class bạn muốn xem, có thể thay đổi
targets = [ClassifierOutputTarget(1)]


for root, _, files in os.walk(args.data_dir):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_file = os.path.join(root, file)
        img_pil = Image.open(img_file).convert("RGB")

        visualization, grayscale_cam = visualize_grad_cam(
            img_pil, grad_cam, targets)

        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(os.path.join(
            args.output_dir, f"grad_cam_{file}"), bbox_inches='tight', pad_inches=0.1)
