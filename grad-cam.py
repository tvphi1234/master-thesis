import os
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import CustomModel
from utils import visualize_grad_cam, DEVICE, CLASS_NAMES
from dataloader import get_dataloader, get_val_transforms


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
                                 'repvgg_a0', 'mobilenetv2_100'],
                        help='Model architecture to use (default: xception)')
    parser.add_argument('--output-dir', type=str, default="./grad_cam_results",
                        help='Directory to save Grad-CAM results (default: ./grad_cam_results)')
    parser.add_argument('--task-type', type=str, default='cancer',
                        choices=['cancer', 'stage', 'multi_task'],
                        help='Task type for the model (default: cancer)')
    return parser.parse_args()


def get_layer_name(model, model_name):
    # Access layers through model.backbone since CustomModel wraps the backbone
    if model_name == "mobilenetv2_100":
        return [model.backbone.bn2.act]
    elif model_name == "resnet50":
        return [model.backbone.layer4[2].act3]
    elif model_name == "xception":
        return [model.backbone.act4]
    elif model_name == "repvgg_a0":
        return [model.backbone.final_conv]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":

    # Parse command line arguments
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = CustomModel(model_name=args.model,
                        task_type=args.task_type).to(DEVICE)
    model.load_pretrained_weights(args.model_path)
    model.eval()


    # Chọn layer cuối cùng của backbone (tùy mô hình, với Xception thường là 'block14_sepconv2')
    target_layers = get_layer_name(model, args.model)

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

            results = model.predict_single_image(img_pil)
            print(file, results)

            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(os.path.join(
                args.output_dir, f"grad_cam_{file}"), bbox_inches='tight', pad_inches=0.1)
            
            # copy original image to output dir
            img_pil.save(os.path.join(
                args.output_dir, f"original_{file}"))
