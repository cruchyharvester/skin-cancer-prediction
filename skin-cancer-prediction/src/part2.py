# part2.py - Load, preprocess, split, and save HAM10000 data to HDF5 safely

import os
import cv2
import numpy as np
import pandas as pd
import h5py
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ✅ Constants
IMG_SIZE = 224
BATCH_SIZE = 500  # process 500 images at a time
DATASET_PATHS = [
    r"C:\Users\mahir\OneDrive - vit.ac.in\skin C\HAM10000_images_part_1",
    r"C:\Users\mahir\OneDrive - vit.ac.in\skin C\HAM10000_images_part_2"
]
METADATA_PATH = r"C:\Users\mahir\OneDrive - vit.ac.in\skin C\HAM10000_metadata.csv"

print("="*60)
print("📊 Loading metadata...")

# ✅ Load Metadata
metadata = pd.read_csv(METADATA_PATH)
imageid_to_label = dict(zip(metadata['image_id'], metadata['dx']))
class_labels = sorted(metadata['dx'].unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}

# ✅ Collect image paths + labels
image_paths, labels = [], []
for folder in DATASET_PATHS:
    for img_file in os.listdir(folder):
        if img_file.endswith(".jpg"):
            img_id = img_file[:-4]
            if img_id in imageid_to_label:
                image_paths.append(os.path.join(folder, img_file))
                labels.append(label_to_index[imageid_to_label[img_id]])

print(f"✅ Total valid images: {len(image_paths)}")

# ✅ Convert labels to one-hot
y = to_categorical(labels, num_classes=len(class_labels))

# ✅ Split into train, val, test
idx_temp, idx_test = train_test_split(
    np.arange(len(image_paths)), test_size=0.2, stratify=y, random_state=42
)
idx_train, idx_val = train_test_split(
    idx_temp, test_size=0.1, stratify=y[idx_temp], random_state=42
)

splits = {
    "train_data.h5": idx_train,
    "val_data.h5": idx_val,
    "test_data.h5": idx_test
}

# ✅ Save in HDF5 safely
def save_hdf5(filename, indices):
    print(f"\n💾 Writing {filename} ...")
    total = len(indices)
    with h5py.File(filename, "w") as f:
        X_dset = f.create_dataset(
            "X", shape=(total, IMG_SIZE, IMG_SIZE, 3),
            dtype='float32', compression="gzip",
            chunks=(min(BATCH_SIZE, total), IMG_SIZE, IMG_SIZE, 3)
        )
        y_dset = f.create_dataset(
            "y", shape=(total, len(class_labels)),
            dtype='float32', compression="gzip"
        )
        
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_idx = indices[start:end]

            X_batch, y_batch = [], []
            for i in batch_idx:
                img = cv2.imread(image_paths[i])
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype('float32') / 255.0
                    X_batch.append(img)
                    y_batch.append(y[i])

            if X_batch:
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                X_dset[start:start+len(X_batch)] = X_batch
                y_dset[start:start+len(y_batch)] = y_batch

            print(f"  ✓ {end}/{total} written")

    print(f"✅ Saved {filename}")

# ✅ Save splits
for fname, idxs in splits.items():
    save_hdf5(fname, idxs)

# ✅ Save class labels
with open("class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)

print("\n" + "="*60)
print("✅ All splits saved successfully at 224x224 resolution")
print("📋 Classes:", class_labels)
