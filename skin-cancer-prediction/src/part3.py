# part3.py - Load from HDF5 and define memory-efficient generator

import h5py
import pickle
import numpy as np
from tensorflow.keras.utils import Sequence

# ✅ Constants
SEQUENCE_LENGTH = 4
IMG_SIZE = 224
BATCH_SIZE = 8

# ✅ Load data from HDF5
def load_hdf5(file):
    with h5py.File(file, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
    return X, y

X_train, y_train = load_hdf5("train_data.h5")
X_val, y_val = load_hdf5("val_data.h5")

# ✅ Load class labels
with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# ✅ Data Generator
class SkinSequence(Sequence):
    def __init__(self, images, labels, batch_size=BATCH_SIZE, seq_len=SEQUENCE_LENGTH, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch, y_batch = [], []
        for i in batch_indices:
            img = self.images[i]
            # ✅ Repeat the full 224x224 image seq_len times to create a sequence
            sequence = np.repeat(img[np.newaxis, ...], self.seq_len, axis=0)
            X_batch.append(sequence)
            y_batch.append(self.labels[i])
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ✅ Instantiate generators
train_generator = SkinSequence(X_train, y_train)
val_generator = SkinSequence(X_val, y_val)

print("✅ Generators created and ready for training with shape (batch, 4, 224, 224, 3).")
