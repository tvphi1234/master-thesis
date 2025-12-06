import os
import matplotlib.pyplot as plt

from collections import defaultdict


def analyze_dataset(dataset_dir):
    class_counts = {}
    for root, dirs, files in os.walk(dataset_dir):
        if len(files) == 0:
            continue

        # Skip directories that are not classes
        class_name = os.path.basename(root)
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += len(files)

    # Print summary
    total_images = sum(class_counts.values())
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {total_images}\n")
    for class_name, count in class_counts.items():
        print(f"Class '{class_name}': {count} images")

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Image Count per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(dataset_dir, "class_distribution.png"))


def extract_patient_id(filename):
    id = filename.replace(".png", "").split(" ")[0]
    return id


def analyze_patients(dataset_dir):
    # patient_images[class][patient_id] = count
    patient_images = defaultdict(lambda: defaultdict(int))
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            class_name = os.path.basename(root)
            patient_id = extract_patient_id(fname)

            patient_images[class_name][patient_id] += 1

    class_names = sorted(patient_images.keys())

    # Lấy danh sách tất cả bệnh nhân
    all_patients = sorted(
        set(pid for class_dict in patient_images.values() for pid in class_dict.keys()))

    # Chuẩn bị dữ liệu cho stacked bar chart
    data_per_class = []
    for class_name in class_names:
        data = [patient_images[class_name].get(pid, 0) for pid in all_patients]
        data_per_class.append(data)

    # Vẽ stacked bar chart
    plt.figure(figsize=(max(10, len(all_patients)//5), 6))
    bottom = [0] * len(all_patients)
    colors = ['#1f77b4', '#ff7f0e']  # Thay đổi nếu có nhiều class hơn
    for idx, (class_name, data) in enumerate(zip(class_names, data_per_class)):
        plt.bar(all_patients, data, bottom=bottom,
                label=class_name, color=colors[idx % len(colors)])
        bottom = [b + d for b, d in zip(bottom, data)]

    # Tổng số hình ảnh cho toàn bộ
    total_images = sum(sum(data) for data in data_per_class)
    print(f"Tổng số hình ảnh: {total_images}")

    # Tổng số bệnh nhân
    total_patients = len(all_patients)
    print(f"Tổng số bệnh nhân: {total_patients}")

    # Số lượng ảnh cho class 0
    class_0_count = sum(patient_images[class_names[0]].values())
    print(f"Số lượng ảnh cho class '{class_names[0]}': {class_0_count}")

    # Số lượng ảnh cho class 1
    class_1_count = sum(patient_images[class_names[1]].values())
    print(f"Số lượng ảnh cho class '{class_names[1]}': {class_1_count}")

    plt.xlabel('Bệnh nhân (ID)')
    plt.ylabel('Số lượng hình')
    plt.title('Số lượng hình ảnh theo từng bệnh nhân và class')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        dataset_dir, "stacked_bar_images_per_patient.png"))


if __name__ == "__main__":
    analyze_dataset("data")
    print("\n---\n")
    analyze_dataset("data/train")
    print("\n---\n")
    analyze_dataset("data/val")

    # print("-------total images in data:")
    # analyze_patients("data_patients")
    # print("\n-----data_patients/train\n")
    # analyze_patients("data_patients/train")
    # print("\n-----data_patients/val\n")
    # analyze_patients("data_patients/val")
