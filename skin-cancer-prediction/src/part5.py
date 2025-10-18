import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ✅ Constants
SEQUENCE_LENGTH = 4
IMG_SIZE = 224
MODEL_PATH = "skin_cancer_cnn_lstm_model.keras"

# ✅ Load Test Data
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# ✅ Convert to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# ✅ Resize each test image to (224, 224, 3)
X_test_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_test])

# ✅ Prepare 4-frame sequences (LSTM expects sequences)
# If your test images are standalone, we repeat each image 4 times to form a sequence
X_test_seq = np.array([[frame] * SEQUENCE_LENGTH for frame in X_test_resized])

# ✅ One-hot encode labels if not already
if y_test.ndim == 1:
    y_test = to_categorical(y_test, num_classes=len(class_labels))

# ✅ Shape check
print("X_test_seq shape:", X_test_seq.shape)
print("y_test shape:", y_test.shape)

# ✅ Load Trained Model
model = load_model(MODEL_PATH)

# ✅ Evaluate the model
loss, accuracy = model.evaluate(X_test_seq, y_test, verbose=1)
print(f"\n✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")

# ✅ Make Predictions
y_pred_probs = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# ✅ Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ✅ Confusion Matrix Visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
