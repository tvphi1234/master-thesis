import os
import shutil
from sklearn.model_selection import train_test_split


def prepare_dataset(dataset_dir, output_dir, train_ratio):
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through each class folder
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get all file paths for the current class
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Split files into train and validation sets
        if train_ratio == 0:
            train_files = []
            val_files = files
        elif train_ratio == 1:
            train_files = files
            val_files = []
        else:
            train_files, val_files = train_test_split(files, train_size=train_ratio, random_state=42)

        # Create class subdirectories in train and val folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy files to the respective directories
        for file in train_files:
            file_name = os.path.dirname(file).split('/')[-2] + '_' + os.path.basename(file)
            shutil.copy(file, os.path.join(train_class_dir, file_name))
        for file in val_files:
            file_name = os.path.dirname(file).split('/')[-2] + '_' + os.path.basename(file)
            shutil.copy(file, os.path.join(val_class_dir, file_name))

        print(f"Class '{class_name}': {len(train_files)} train, {len(val_files)} val")

if __name__ == "__main__":
    # Parameters
    OUTPUT_DIR = "data_patients"  # Path to save the split dataset
    data_dict = {
        "dataset/x10/04.06.2025": 1.0,
        "dataset/x10/17.06.2025": 1.0,
        "dataset/x10/26.06.2025": 1.0,
        "dataset/x10/11.07.2025": 1.0,
        "dataset/x10/16.07.2025": 1.0,
        "dataset/x10/23.07.2025": 1.0,
        "dataset/x10/29.07.2025": 1.0,
        "dataset/x10/05.08.2025": 1.0,
        "dataset/x10/19.08.2025": 0.0
    }

    for DATASET_DIR, TRAIN_RATIO in data_dict.items():
        prepare_dataset(DATASET_DIR, OUTPUT_DIR, TRAIN_RATIO)
        print("Dataset preparation complete!")
