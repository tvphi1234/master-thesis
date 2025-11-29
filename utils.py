import cv2
import torch
import numpy as np

from PIL import Image
from timm import create_model
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image


IMG_SIZE = 640
MAX_HEIGHT = 860
MAX_WIDTH = 1240
# "mobilenetv2_100", "resnet50", "xception", "repvgg_a0"
MODEL_NAME = "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Benign", "Cancer"]  # 0: Benign, 1: Cancer


def load_model(model_name=MODEL_NAME, num_classes=2, model_path=None, is_train=True):
    # Load model
    model = create_model(model_name, pretrained=False, num_classes=num_classes)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model = model.to(DEVICE)

    if not is_train:
        model.eval()

    return model


def get_train_transforms():
    # Data Preparation
    transform_train = transforms.Compose([
        # PadToMaxSize(),  # Add padding to cover the maximum size
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.3),    # Random vertical flip
        # Random rotation (-30 to +30 degrees)
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    return transform_train


def get_val_transforms():
    transform_val = transforms.Compose([
        # PadToMaxSize(),  # Add padding to cover the maximum size
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    return transform_val


def reverse_orginal_size(img, original_size):
    # zero img
    zero_img = np.zeros(
        (original_size[0], original_size[1], 3), dtype=np.uint8)

    # get padding
    (pad_left, pad_top, pad_right,
     pad_bottom) = PadToMaxSize().get_padding(original_size)

    # resize img to PadToMaxSize
    img = cv2.resize(img, (MAX_WIDTH, MAX_HEIGHT))

    # remove padding
    img = img[pad_top:MAX_HEIGHT - pad_bottom, pad_left:MAX_WIDTH - pad_right]
    return img


def visualize_grad_cam(img_pil, grad_cam, targets):
    original_size = img_pil.size  # (width, height)

    # val transform
    val_transform = get_val_transforms()

    # transform image
    input_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

    # grad-CAM
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)[0]

    # # reverse to original size
    # grayscale_cam = reverse_orginal_size(grayscale_cam, original_size)

    grayscale_cam = cv2.resize(
        grayscale_cam, (original_size[0], original_size[1]))

    # visualization
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    visualization = Image.fromarray(visualization)

    return visualization, grayscale_cam


def model_predict(model, img_pil):
    """Make prediction on the image"""

    # val transform
    val_transform = get_val_transforms()

    # transform image
    image_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        return predicted_class.item(), confidence.item(), probabilities[0].cpu().numpy()


# define a function to add padding to images to cover the maximum size
class PadToMaxSize:
    def __init__(self, max_height=MAX_HEIGHT, max_width=MAX_WIDTH):
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, img):
        height, width = img.size
        padding_height = self.max_height - height
        padding_width = self.max_width - width
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        return transforms.functional.pad(img, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

    def get_padding(self, original_size):
        width, height = original_size
        padding_height = self.max_height - height
        padding_width = self.max_width - width
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        return (padding_left, padding_top, padding_right, padding_bottom)
