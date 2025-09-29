import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import DEVICE
from utils import load_model, visualize_grad_cam

# Tham số
IMG_PATH = "./cache/wrong_predict"  # Đường dẫn ảnh sai
MODEL_PATH = "./models/xception_best_20250826.pth"


model = load_model(model_name="xception",
                   num_classes=2,
                   model_path=MODEL_PATH,
                   is_train=False)

# Chọn layer cuối cùng của backbone (tùy mô hình, với Xception thường là 'block14_sepconv2')
target_layers = [model.act4]

# Grad-CAM
grad_cam = GradCAM(model=model, target_layers=target_layers)

# 1 là class bạn muốn xem, có thể thay đổi
targets = [ClassifierOutputTarget(1)]


for root, _, files in os.walk(IMG_PATH):
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
            root, f"grad_cam_{file}"), bbox_inches='tight', pad_inches=0.1)
