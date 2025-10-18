# Skin Cancer Detection using CNN + LSTM (MobileNetV2)

This project detects different types of skin cancer from dermoscopic images using a deep learning hybrid architecture that combines **MobileNetV2 (CNN)** for spatial feature extraction and **LSTM** for sequential pattern learning.  
It aims to create an accurate and efficient model suitable for real-world deployment in teledermatology and healthcare applications.

---

## Features
-  Hybrid **CNN + LSTM** model for advanced feature learning  
-  **Transfer Learning** using pretrained MobileNetV2  
-  Handles **class imbalance** with Focal Loss  
-  Includes visualizations: accuracy/loss curves, confusion matrix, and sample predictions  
-  Modular and reusable code (training, evaluation, inference)  
-  Ready for integration with web apps (Streamlit/Flask)

---

##  Dataset
- **Dataset Name:** [HAM10000 (Human Against Machine with 10000 training images)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Classes:** 7 types of skin lesions  
  *(nv, mel, bkl, bcc, akiec, vasc, df)*  
- **Data Split:** Train / Validation / Test

>  The dataset and model weights are **not included** in this repository due to size limits.  
> Download the dataset manually from Kaggle and place it under a folder named `data/`.

