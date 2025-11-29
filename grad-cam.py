import os
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import MODEL_NAME
from utils import load_model, visualize_grad_cam, model_predict

# Tham số
DATA_SET_PATH = "cache/wrong_repvgg_a0"  # Đường dẫn ảnh sai
MODEL_PATH = "pretrained_models/models_x10/2025_11_21_640x640/resnet50_20251120_best.pth"
OUTDIR = "./a"

os.makedirs(OUTDIR, exist_ok=True)

model = load_model(model_name=MODEL_NAME,
                   num_classes=2,
                   model_path=MODEL_PATH,
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


for root, _, files in os.walk(DATA_SET_PATH):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_file = os.path.join(root, file)
        img_pil = Image.open(img_file).convert("RGB")

        visualization, grayscale_cam = visualize_grad_cam(
            img_pil, grad_cam, targets)

        results = model_predict(model, img_pil)
        print(file, results)

        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(os.path.join(
            OUTDIR, f"grad_cam_{file}"), bbox_inches='tight', pad_inches=0.1)
