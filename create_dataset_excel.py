import os
import pandas as pd
from pathlib import Path


def create_dataset_excel():
    """
    Create an Excel file with image names and their cancer classification
    from the training dataset folders.
    """
    # Define the paths
    train_path = "/home/cybercore/Workspaces/phitv/classify/data/train"
    benign_path = os.path.join(train_path, "Benign")
    cancer_path = os.path.join(train_path, "Cancer")

    # Initialize lists to store data
    image_names = []
    cancer_labels = []

    # Read Benign images
    if os.path.exists(benign_path):
        benign_images = [f for f in os.listdir(
            benign_path) if f.lower().endswith('.png')]
        print(f"Found {len(benign_images)} benign images")

        for image in benign_images:
            image_names.append(image)
            cancer_labels.append("No")  # Not cancer

    # Read Cancer images
    if os.path.exists(cancer_path):
        cancer_images = [f for f in os.listdir(
            cancer_path) if f.lower().endswith('.png')]
        print(f"Found {len(cancer_images)} cancer images")

        for image in cancer_images:
            image_names.append(image)
            cancer_labels.append("Yes")  # Is cancer

    # Create DataFrame
    df = pd.DataFrame({
        'Image Name': image_names,
        'Cancer': cancer_labels
    })

    # Sort by image name for better organization
    df = df.sort_values('Image Name').reset_index(drop=True)

    # Save to Excel file
    output_file = "/home/cybercore/Workspaces/phitv/classify/dataset_classification.xlsx"
    df.to_excel(output_file, index=False)

    print(f"\nDataset Excel file created successfully!")
    print(f"File saved as: {output_file}")
    print(f"Total images: {len(df)}")
    print(f"Benign images: {len(df[df['Cancer'] == 'No'])}")
    print(f"Cancer images: {len(df[df['Cancer'] == 'Yes'])}")

    # Display first few rows
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10))

    return df


if __name__ == "__main__":
    dataset_df = create_dataset_excel()
