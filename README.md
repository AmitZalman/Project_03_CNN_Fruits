# 🍎 Fruit CNN Classifier

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify images of fruits and vegetables into 5 categories. Includes a full training pipeline in Google Colab and a live interactive web app built with Streamlit.

---

## 🎯 Project Overview

This project trains a CNN from scratch to recognize 5 types of produce:

| Class | Training Images | Test Images |
|---|---|---|
| 🍎 Apple | ~933 | ~146 |
| 🍌 Banana | ~495 | ~166 |
| 🥒 Cucumber | ~860 | ~50 |
| 🍇 Grape | ~965 | ~155 |
| 🍓 Strawberry | ~1,186 | ~164 |
| **Total** | **4,438** | **681** |

The trained model is served through a **Streamlit web app** that allows users to upload a photo or use their camera, and get an instant prediction with confidence score.

---

## 🏗️ Architecture

```
Input (100×100×3)
      │
      ▼
┌─────────────────────────────┐
│   Data Augmentation         │
│   RandomFlip (horizontal)   │
│   RandomRotation (±10%)     │
│   RandomZoom (±10%)         │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Conv2D(32, 3×3) + ReLU     │
│  MaxPooling2D(2×2)          │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Conv2D(64, 3×3) + ReLU     │
│  MaxPooling2D(2×2)          │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Conv2D(128, 3×3) + ReLU    │
│  MaxPooling2D(2×2)          │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Flatten                    │
│  Dense(128) + ReLU          │
│  Dropout(0.5)               │
│  Dense(5) + Softmax         │
└─────────────────────────────┘
      │
      ▼
   5 Classes
```

**Total Parameters:** ~1,732,421 (6.61 MB)

---

## 📊 Training Results

Trained for **10 epochs** on Google Colab with a GPU.

| Metric | Value |
|---|---|
| Final Training Accuracy | ~95.5% |
| Final Validation Accuracy | **100%** |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

The model uses **Data Augmentation** and **Dropout (0.5)** to reduce overfitting.

---

## 📁 Project Structure

```
Project_03_CNN_Fruits/
│
├── Project_03.ipynb          # Google Colab training notebook
├── app.py                    # Streamlit web application
├── my_fruits_model.keras     # Trained CNN model weights
├── requirements.txt          # Python dependencies
├── saved_images/             # Folder for images saved via the app
└── README.md
```

---

## 🚀 Running the Web App

### 1. Clone the repository

```bash
git clone https://github.com/amitzalman/Project_03_CNN_Fruits.git
cd Project_03_CNN_Fruits
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Web App Features

- **📷 Camera input** — take a photo directly from your device's camera
- **📁 File upload** — upload any JPG / PNG / WEBP image
- **🧠 Instant prediction** — shows the predicted fruit class and confidence score
- **⚠️ Confidence threshold** — if confidence is below 60%, the app returns "Unknown" instead of a wrong guess
- **💾 Save image** — save any uploaded image to the local `saved_images/` folder

---

## 🔧 Training (Google Colab)

The full training pipeline is available in `Project_03.ipynb`:

1. Mount Google Drive and extract the dataset ZIP
2. Load images using `tf.keras.utils.image_dataset_from_directory`
3. Normalize folder names (strip trailing spaces)
4. Build the CNN model with augmentation and dropout
5. Compile with Adam optimizer
6. Train for 10 epochs with validation
7. Plot accuracy and loss curves
8. Test single-image prediction
9. Save the model as `.keras`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| TensorFlow / Keras | Model training & inference |
| Streamlit | Interactive web app |
| NumPy | Array manipulation |
| Pillow (PIL) | Image loading & preprocessing |
| Matplotlib | Training plots |
| Google Colab | GPU-accelerated training environment |

---

## 📝 Notes

- Images are resized to **100×100 pixels** before feeding into the model — this matches the training resolution.
- No pixel normalization (divide by 255) was applied during training, so the app also skips it.
- The dataset originates from a subset of the [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) dataset on Kaggle.

---

## 👤 Author

**Amit Zalman**  
GitHub: [@amitzalman](https://github.com/amitzalman)
