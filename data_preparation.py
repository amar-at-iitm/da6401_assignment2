import os
import shutil
import zipfile
import random
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def rename_val_to_test(data_dir):
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')
    if os.path.exists(val_path):
        os.rename(val_path, test_path)
        print(f"Renamed 'val' to 'test'")
    else:
        print("No 'val' folder found to rename.")

def create_val_split(train_dir, val_dir, split_ratio=0.2):
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = os.listdir(class_path)
        num_val = int(len(images) * split_ratio)
        val_images = random.sample(images, num_val)

        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)

        for img_name in val_images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(val_class_dir, img_name)
            shutil.move(src, dst)

        print(f"[{class_name}] -> Moved {num_val} images to validation set.")

if __name__ == "__main__":
    # Configurations
    dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    zip_filename = "nature_12K.zip"
    extracted_dir = "inaturalist_12K"

    # Step 1: Downloading
    zip_path = zip_filename
    if not os.path.exists(zip_path):
        download_file(dataset_url, zip_path)
    else:
        print("Dataset already downloaded.")

    # Step 2: Unzipping
    if not os.path.exists(extracted_dir):
        unzip_file(zip_path, ".")
    else:
        print("Dataset already unzipped.")

    # Step 3: Renaming val -> test
    rename_val_to_test(os.path.join(extracted_dir))

    # Step 4: Creating new validation split from train/
    train_path = os.path.join(extracted_dir, "train")
    val_path = os.path.join(extracted_dir, "val")
    create_val_split(train_path, val_path, split_ratio=0.2)
