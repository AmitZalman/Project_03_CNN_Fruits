# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
from PIL import Image
from keras.models import load_model

# Set page configuration (title and icon)
st.set_page_config(page_title="Fruits CNN Predictor", page_icon="🍎")

st.title("🍎 Fruits CNN Predictor")
st.write("Take a photo or upload one, and the model will identify the fruit!")

# ---------- Settings ----------
# The name of the model file you saved and downloaded from Colab
MODEL_PATH = "my_fruits_model.keras"

# Image size must match exactly what the model was trained on
IMG_SIZE = (100, 100)

# Class names - Ensure these match exactly the folder names from your training data
CLASS_NAMES = ['Apple', 'Banana', 'Cucumber', 'Grape', 'Strawberry']

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_model():
    # Load the trained CNN model
    return load_model(MODEL_PATH)


model = get_model()


def save_uploaded_image(file_obj, prefix: str = "fruit") -> Path:
    # Extract original extension or default to .jpg
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    # Generate a unique filename using timestamp and uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    # Save the file to the local directory
    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path


def preprocess_for_model(uploaded_file) -> np.ndarray:
    """
    Converts Streamlit uploaded image to a model-ready numpy array.
    This matches the preprocessing done during training.
    """
    # Streamlit UploadedFile -> bytes -> PIL Image
    img = Image.open(uploaded_file)

    # Ensure 3 channels (RGB) - handles PNG with alpha (RGBA)
    img = img.convert("RGB")

    # Resize to model expected input size (100x100)
    img = img.resize(IMG_SIZE)

    # Convert PIL image to numpy array
    arr = np.array(img, dtype=np.float32)

    # Note: We do NOT divide by 255.0 because we didn't normalize during training

    # Add batch dimension -> Output shape becomes (1, 100, 100, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------- UI ----------
st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Predict Fruit!"):
            try:
                # Preprocess the image
                x = preprocess_for_model(chosen)

                # Run the model
                predictions = model.predict(x, verbose=0)[0]

                # Extract the index and name of the predicted class
                predicted_class_index = np.argmax(predictions)
                confidence = np.max(predictions) * 100

                # הוספת תנאי הבונוס: בדיקה האם רמת הביטחון גבוהה מ-60%
                if confidence < 60.0:
                    st.warning("Prediction: **unknown**")
                    st.info(f"Confidence was too low: {confidence:.2f}%")
                else:
                    predicted_fruit = CLASS_NAMES[predicted_class_index]
                    # Display the result to the user
                    st.success(f"Prediction: **{predicted_fruit}**")
                    st.info(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        # ===================================

    with col2:
        if st.button("💾 Save image to computer"):
            try:
                saved_path = save_uploaded_image(chosen, prefix="fruit")
                st.success(f"Saved ✅  {saved_path.resolve()}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")
st.caption(f"Model file: {Path(MODEL_PATH).resolve()}")