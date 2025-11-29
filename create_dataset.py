import os
import csv
import shutil

from sklearn.model_selection import train_test_split


def split_dataset(annotation_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, field_split="level"):
    """ Split dataset into train, val, and test sets based on given ratios. """

    with open(annotation_file, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        image_paths = [row['image_path'] for row in reader]

    if field_split == "image_path":
        train_paths, temp_paths = train_test_split(
            image_paths, train_size=train_ratio, random_state=42)
        val_paths, test_paths = train_test_split(
            temp_paths, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    elif field_split == "level":
        levels = {}
        with open(annotation_file, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                level = row['level']
                if level not in levels:
                    levels[level] = []
                levels[level].append(row['image_path'])

        train_paths, val_paths, test_paths = [], [], []
        for level, paths in levels.items():
            train_level, temp_level = train_test_split(
                paths, train_size=train_ratio, random_state=42)
            val_level, test_level = train_test_split(
                temp_level, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

            train_paths.extend(train_level)
            val_paths.extend(val_level)
            test_paths.extend(test_level)

        # write back to new annotation files
        def write_split_file(split_paths, split_name):
            split_file = os.path.join(os.path.dirname(
                annotation_file), f"{split_name}_annotations.csv")
            with open(split_file, mode='w', newline='') as csv_file:
                fieldnames = ['image_path', 'patient_id', 'class', 'level']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for path in split_paths:
                    with open(annotation_file, mode='r') as original_file:
                        original_reader = csv.DictReader(original_file)
                        for row in original_reader:
                            if row['image_path'] == path:
                                writer.writerow(row)
                                break

        write_split_file(train_paths, "train")
        write_split_file(val_paths, "val")
        write_split_file(test_paths, "test")

    return train_paths, val_paths, test_paths


def create_annotations_file(data_dict, output_dir="data"):
    """ Create a CSV file with image paths and labels for the dataset. """

    annotations_file = os.path.join(output_dir, "annotations.csv")
    with open(annotations_file, mode='w', newline='') as csv_file:
        fieldnames = ['image_path', 'patient_id', 'class', 'level']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_dir in data_dict.keys():
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    # Determine class
                    class_name = "Benign" if "Benign" in root else "Cancer"

                    # Determine patient ID
                    file_names = file.split(' (')
                    if len(file_names) > 1:
                        patient_id = file_names[0].strip()
                    else:
                        patient_id = "Unknown"

                    # Determine level based on folder structure
                    if class_name == "Benign":
                        level = 0
                    elif 'level_1' in root:
                        level = 1
                    elif 'level_2' in root:
                        level = 2
                    else:
                        level = -1  # Unknown level

                    image_path = os.path.join(root, file)
                    writer.writerow({
                        'image_path': image_path,
                        'patient_id': patient_id,
                        'class': class_name,
                        'level': level
                    })


if __name__ == "__main__":
    # Parameters
    OUTPUT_DIR = "data"  # Path to save the split dataset
    # data_dict = {
    #     "dataset/x10/04.06.2025": 0.8,
    #     "dataset/x10/17.06.2025": 0.8,
    #     "dataset/x10/26.06.2025": 0.8,
    #     "dataset/x10/11.07.2025": 0.8,
    #     "dataset/x10/16.07.2025": 0.8,
    #     "dataset/x10/23.07.2025": 0.8,
    #     "dataset/x10/29.07.2025": 0.8,
    #     "dataset/x10/05.08.2025": 0.8,
    #     "dataset/x10/19.08.2025": 0.8,
    #     "dataset/x10/24.09.2025": 0.8,
    #     "dataset/x10/02.10.2025": 0.8
    # }
    # create_annotations_file(data_dict)
    # print("Annotations file creation complete!")

    ANNOTATION_FILE = os.path.join(OUTPUT_DIR, "annotations.csv")
    train_paths, val_paths, test_paths = split_dataset(
        ANNOTATION_FILE, field_split="level")
