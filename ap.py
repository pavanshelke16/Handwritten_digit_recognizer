import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the trained model
model = models.load_model('my_model.keras')

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_resized = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img_resized)  # Convert to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array

def main():
    st.title("Handwritten Digit Recognizer")
    st.markdown("<p style='margin-top:-15px; margin-bottom:5px; font-size: 10px;'>by PAVAN SHELKE</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with half of its original width
        st.image(uploaded_file, caption="Uploaded Image.", width=uploaded_file.type.width // 2 if hasattr(uploaded_file.type, 'width') else None)

        # Save the uploaded file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Add a "Predict" button
        if st.button("Predict"):
            # Preprocess the uploaded image
            new_image = preprocess_image(file_path)

            # Make predictions using the model
            predictions = model.predict(new_image)
            predicted_digit = np.argmax(predictions[0])

            # Display the predicted digit in a larger font
            st.header(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
