import os
import re
import torch

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from collections import defaultdict

IMG_SIZE = 299
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "xception_model.pth"
VAL_DIR = "data/val"

def extract_patient_id(filename):
    id = filename.replace(".png", "").split(" ")[0]
    return id

# Data transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Đánh giá theo bệnh nhân
patient_correct = defaultdict(int)
patient_total = defaultdict(int)

for i, (inputs, labels) in enumerate(val_loader):
    # Lấy đường dẫn file ảnh gốc
    path, _ = val_dataset.samples[i]
    patient_id = extract_patient_id(os.path.basename(path))
    if patient_id is None:
        continue

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if preds.item() == labels.item():
            patient_correct[patient_id] += 1
        patient_total[patient_id] += 1

# In kết quả
print("Độ chính xác theo từng bệnh nhân:")
patient_ids = []
accuracies = []
for pid in sorted(patient_total.keys()):
    acc = patient_correct[pid] / patient_total[pid] * 100
    print(f"Bệnh nhân {pid}: {acc:.2f}% ({patient_correct[pid]}/{patient_total[pid]})")



patient_ids = sorted(patient_total.keys())
correct_counts = [patient_correct[pid] for pid in patient_ids]
incorrect_counts = [patient_total[pid] - patient_correct[pid] for pid in patient_ids]

plt.figure(figsize=(max(10, len(patient_ids)//5), 6))
plt.bar(patient_ids, correct_counts, color='limegreen', label='Đúng')
plt.bar(patient_ids, incorrect_counts, bottom=correct_counts, color='red', label='Sai')
plt.xlabel('Bệnh nhân (ID)')
plt.ylabel('Số lượng hình đánh giá')
plt.title('Số lượng hình đúng/sai theo từng bệnh nhân')
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig("patient_stack_accuracy_bar.png")
