import os
import csv
import time
import shutil

from sklearn.model_selection import train_test_split


def split_dataset(annotation_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, field_split="image_path", magnification="x10"):
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

    write_split_file(train_paths, f"{magnification}_train")
    write_split_file(val_paths, f"{magnification}_val")
    write_split_file(test_paths, f"{magnification}_test")

    return train_paths, val_paths, test_paths


def split_dataset_from_Warwick_dataset(image_dir, annotation_file):
    """ Split dataset from Warwick into train, val, and test sets. """

    with open(annotation_file, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)

        patient_ids, image_paths, classes = [], [], []
        for row in reader:
            image_paths.append(row['name'])
            patient_ids.append(row['patient ID'])
            classes.append(row[' grade (GlaS)'])

    train_file = os.path.join(os.path.dirname(
        annotation_file), f"warwick_train_annotations.csv")
    test_file = os.path.join(os.path.dirname(
        annotation_file), f"warwick_test_annotations.csv")

    def write_split_file(split_file, is_train=True):

        with open(split_file, mode='w', newline='') as csv_file:
            fieldnames = ['image_path', 'patient_id', 'class', 'level']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for image_path, patient_id, cls in zip(image_paths, patient_ids, classes):
                if is_train and "train" in image_path:
                    writer.writerow({
                        'image_path': os.path.join(image_dir, image_path) + ".bmp",
                        'patient_id': patient_id,
                        'class': "Benign" if cls == " benign" else "Cancer",
                        'level': -1
                    })
                elif not is_train and "test" in image_path:
                    writer.writerow({
                        'image_path': os.path.join(image_dir, image_path) + ".bmp",
                        'patient_id': patient_id,
                        'class': "Benign" if cls == " benign" else "Cancer",
                        'level': -1
                    })

    write_split_file(train_file, is_train=True)
    write_split_file(test_file, is_train=False)


def create_annotations_file(data_dict, output_dir="data", ann_file_name="annotations.csv"):
    """ Create a CSV file with image paths and labels for the dataset. """

    annotations_file = os.path.join(output_dir, ann_file_name)
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
                    elif 'stage_1' in root:
                        level = 1
                    elif 'stage_2' in root:
                        level = 2
                    elif 'stage_3' in root:
                        level = 3
                    elif 'unknown' in root:
                        level = -1  # Unknown level

                    image_path = os.path.join(root, file)
                    writer.writerow({
                        'image_path': image_path,
                        'patient_id': patient_id,
                        'class': class_name,
                        'level': level
                    })

def read_and_sort_annotations(annotation_file):
    """ Read annotations from CSV and return sorted list of image paths. """

    root_dir = "dataset/x10"

    with open(annotation_file, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)

        image_paths = []
        cancer_or_benign = []
        cancer_stage = []
        for row in reader:
            image_paths.append(row['Tên ảnh'])
            cancer_or_benign.append(row['Ung thư hay không'])
            cancer_stage.append(row['Mức độ ung thư'])

        for image_path, is_cancer, stage in zip(image_paths, cancer_or_benign, cancer_stage):
            folder, file_name = image_path.split("_")
            folder = folder.split("/")[-1]

            id, ext = file_name, ""
            if " " in file_name:
                id, ext = file_name.split(" ")

            # create a complete path
            if is_cancer == "No":
                folder_name = "Benign"
            else:
                folder_name = "Cancer"

            if folder_name == "Benign":
                continue
                
            original_path = os.path.join(
                root_dir, folder, folder_name, file_name)
            
            level_folder = ""
            if stage == "1":
                level_folder = "stage_1"
            elif stage == "2":
                level_folder = "stage_2"
            elif stage == "3":
                level_folder = "stage_3"
            elif stage.lower() == "không ung thư":
                folder_name = "Benign"
            elif len(stage) == 0 and is_cancer == "Yes":
                level_folder = "unknown"
            elif len(stage) == 0 and is_cancer == "No":
                folder_name = "Benign"
            else:
                print("Unknown stage:", stage)
                exit()


            new_path = os.path.join(
                root_dir, folder, folder_name, level_folder, file_name)
            
            if os.path.exists(new_path):
                # print("File already exists at destination:", new_path)
                continue

            if not os.path.exists(original_path):
                print("File not found:", original_path)
                continue
                
            # move file
            print("Moving:", original_path, " --> ", new_path)
            # if stage.lower() == "không ung thư":
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(original_path, new_path)




            

    return image_paths

def read_and_change_file_name():
    """ Read annotations from CSV and return sorted list of image paths. """

    root_dir = "dataset/x10"

    # bew file
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            folder, file_name = root.split("/")[-2], file
            id, ext = file_name, ""
            if " " in file_name:
                if len(file_name.split(" ")) > 2:
                    id, ext = file_name.split(" ")[1:]
                else:
                    id, ext = file_name.split(" ")

            # create a complete path
            if folder == "Benign":
                folder_name = "Benign"
            else:
                folder_name = "Cancer"

            if len(id) == 1:
                new_id = "0" + id
            elif len(id) == 2:
                new_id = id

            else:
                new_id = id.split(".")[0]

                if len(new_id) == 1:
                    new_id = "0" + new_id

            if len(ext) == 0:
                ext = "(0).png"
            elif ext == ".png":
                ext = "(0).png"

            new_file_name = f"{new_id} {ext}"

            print(file, " --> ", new_file_name)

            # change file name
            shutil.move(os.path.join(root, file), os.path.join(
                root, new_file_name))


if __name__ == "__main__":

    # read_and_sort_annotations("dataset/dataset_classification.csv")

    # read_and_change_file_name()

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
    #     "dataset/x10/02.10.2025": 0.8,
    #     # "dataset/x4/05.12.2025": 0.8,
    #     # "dataset/x4/17.12.2025": 0.8,
    #     # "dataset/x40/private/07.10": 0.8,
    #     # "dataset/x40/private/10.10": 0.8,
    #     # "dataset/x40/private/21.10": 0.8,
    #     # "dataset/x40/private/23.10": 0.8,
    #     # "dataset/x40/public/test": 0.8,
    #     # "dataset/x40/public/train": 0.8
    # }
    # create_annotations_file(data_dict, ann_file_name="x10_annotations.csv")
    # print("Annotations file creation complete!")

    ANNOTATION_FILE = os.path.join(OUTPUT_DIR, "x40_annotations.csv")
    train_paths, val_paths, test_paths = split_dataset(
        ANNOTATION_FILE, field_split="level", magnification="x40")
    print("Dataset splitting complete!")
    print(train_paths, val_paths, test_paths)

    # # split_dataset_from_Warwick_dataset(
    # #     "dataset/x10/warwick_QU/train",
    # #     "dataset/x10/warwick_QU/Grade.csv")
    # # print("Dataset splitting complete!")
