"""Custom data loader utilities for the project.

Provides:
- `CustomImageDataset` : a flexible torch Dataset supporting folder-structure datasets
  and optional CSV annotations.
- `compute_class_weights` : compute per-sample weights for imbalanced datasets.
- `create_data_loader` : helper to build a torch DataLoader from the dataset.

Example:
    from dataloader import CustomImageDataset, create_data_loader

    ds = CustomImageDataset(root_dir='data/train')
    loader = create_data_loader(ds, batch_size=32, shuffle=True)

The dataset will look for subfolders under `root_dir` and treat each subfolder
as a class (like torchvision.datasets.ImageFolder). Alternatively, pass
`annotations_file` to load (relative) paths and labels from a CSV.
"""

import os
import csv
import torch

from PIL import Image
from typing import Callable, List, Optional, Tuple, Dict, Any
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


IMG_SIZE = 640


def default_image_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert('RGB')


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


class CustomImageDataset(Dataset):
    """Flexible image dataset.

    Args:
        root_dir: Folder containing class-subfolders (optional if using CSV)
        annotations_file: CSV file with two columns: path,label (relative to root_dir or absolute)
        classes: Optional list of class names (if None, detected from subfolders or CSV labels)
        transform: torchvision transforms to apply to images
        loader: function path->PIL.Image (defaults to PIL loader)
        extensions: tuple of allowed image file extensions
        return_path: if True, __getitem__ returns the image path as a third value
        return_metadata: if True, returns metadata dict as an extra item
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        annotations_file: Optional[str] = None,
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        loader: Callable = default_image_loader,
        extensions: Tuple[str, ...] = (
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff'),
        return_path: bool = False,
        return_metadata: bool = False,
        task_type: str = 'cancer',
    ):
        super().__init__()
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.loader = loader
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.return_path = return_path
        self.return_metadata = return_metadata

        self.classes = classes or []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int, Dict[str, Any]]] = []
        self.task_type = task_type   # 'cancer' or 'stage'

        if annotations_file:
            self._load_from_csv(annotations_file)

    def _load_from_csv(self, csv_path: str):
        # expected CSV: path,label (header optional). Paths may be absolute or relative to root_dir
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # detect header
        if len(rows) > 0 and any(cell.lower() in ['image_path', 'patient_id', 'class', 'level'] for cell in rows[0]):
            rows = rows[1:]

        labels_seen = set()
        for row in rows:
            # skip empty rows
            if not row:
                continue

            img_path, patient_id, class_label, level = row

            if self.task_type == 'stage':
                if int(level) < 1:
                    # skip samples with no stage label
                    continue

            if not os.path.exists(img_path):
                # skip missing files but keep processing
                continue

            labels_seen.add(class_label)
            self.samples.append(row)

        # build classes from labels_seen
        sorted_labels = sorted([l for l in labels_seen if l is not None])
        self.classes = sorted_labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        conv_samples = []
        for img_path, patient_id, class_label, level in self.samples:
            if class_label is None:
                continue

            idx = self.class_to_idx[class_label]
            conv_samples.append(
                (img_path, idx, {"patient_id": patient_id, "level": level}))
        self.samples = conv_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, meta = self.samples[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)

        level = meta.get("level")
        if level is not None:
            meta["level"] = int(level)

        level = torch.tensor(meta.get("level", -1),
                             dtype=torch.long) - 1  # adjust to 0-based

        return (img, label, level)

    def get_class_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, idx, _ in self.samples:
            cls = self.classes[idx]
            counts[cls] = counts.get(cls, 0) + 1
        return counts


def compute_class_weights(dataset: CustomImageDataset) -> torch.Tensor:
    """Compute per-class weights (inverse frequency) and return a tensor of per-sample weights.

    Returns a 1D tensor with length == len(dataset) containing the weight to be used by
    `torch.utils.data.WeightedRandomSampler`.
    """
    counts = {}
    for _, idx, _ in dataset.samples:
        counts[idx] = counts.get(idx, 0) + 1

    num_samples = len(dataset)
    num_classes = len(dataset.classes)
    class_weights = {cls: num_samples /
                     (num_classes * cnt) for cls, cnt in counts.items()}

    sample_weights = [class_weights[idx] for _, idx, _ in dataset.samples]
    return torch.tensor(sample_weights, dtype=torch.double)


def get_dataloader(data_dir, annotation_file,
                   data_transform=None, is_shuffle=True, batch_size=1, task_type='cancer'):

    annotation_file = os.path.join(data_dir, annotation_file)

    custom_dataset = CustomImageDataset(
        root_dir=data_dir,
        annotations_file=annotation_file,
        transform=data_transform,
        return_metadata=True,
        task_type=task_type
    )

    data_loader = DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
    )

    return data_loader
