import os
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import load_model, visualize_grad_cam


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


# Chọn layer cuối cùng của backbone (tùy mô hình, với Xception thường là 'block14_sepconv2')
# repvgg_a0: model.final_conv
# resnet 50: model.layer4[2].act3
# exception: model.act4
# mobilenetv2: model.bn2.act

if MODEL_NAME == "mobilenetv2_100":
    target_layers = [model.bn2.act]
elif MODEL_NAME == "resnet50":
    target_layers = [model.layer4[2].act3]
elif MODEL_NAME == "xception":
    target_layers = [model.act4]
elif MODEL_NAME == "repvgg_a0":
    target_layers = [model.final_conv]

# Grad-CAM
grad_cam = GradCAM(model=model, target_layers=target_layers)

# 1 là class bạn muốn xem, có thể thay đổi
targets = [ClassifierOutputTarget(1)]

classes = {
    "Benign": 0,
    "Cancer": 1
}


for root, _, files in os.walk(args.data_dir):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        class_idx = classes[os.path.basename(root)]
        targets = [ClassifierOutputTarget(class_idx)]

        img_file = os.path.join(root, file)
        img_pil = Image.open(img_file).convert("RGB")

        visualization, grayscale_cam = visualize_grad_cam(
            img_pil, grad_cam, targets)

        results = model_predict(model, img_pil)
        print(file, results)

        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(os.path.join(
            args.output_dir, f"grad_cam_{file}"), bbox_inches='tight', pad_inches=0.1)
