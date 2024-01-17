import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load the digit recognizer model
model = load_model('my_model.keras')

def preprocess_image(image):
    # Convert to grayscale
    img_gray = image.convert('L')

    # Resize to 28x28 pixels
    img_resized = img_gray.resize((28, 28))

    # Convert the resized image to a NumPy array
    img_array = np.array(img_resized)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape the image to match the input shape of the model
    img_array = img_array.reshape(1, 28, 28, 1)  # Add an additional dimension for the channel

    return img_array

def main():
    st.title("Handwritten Digit Recognizer")
    st.markdown("<p style='margin-top:-15px; margin-bottom:5px; font-size: 10px;'>by PAVAN SHELKE</p>", unsafe_allow_html=True)

    # Create a canvas for drawing (double the size)
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=5,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Add a "Predict" button
    if st.button("Predict"):
        # Convert the drawn image to a PIL Image
        image = Image.fromarray(canvas_result.image_data)

        # Preprocess the drawn image
        new_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(new_image)
        predicted_digit = np.argmax(predictions[0])

        # Display the predicted digit
        st.header(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
