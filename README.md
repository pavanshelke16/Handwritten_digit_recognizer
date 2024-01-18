# Handwritten Digit Recognizer

This project implements a Handwritten Digit Recognition system using a Neural Network trained on the MNIST dataset. The goal is to accurately predict the digit (0-9) from either a scanned image of a handwritten numeral or a hand-drawn digit on a canvas.

## Dependencies

- TensorFlow
- Streamlit
- NumPy
- PIL (Python Imaging Library)

## Project Overview

The model is developed using the TensorFlow library and trained on the MNIST dataset, which consists of a large collection of 28x28 pixel grayscale images of handwritten digits. The implemented neural network can accurately classify and predict the digits present in scanned images or hand-drawn digits on a canvas.

## Model Architecture

The neural network model consists of a Flatten layer to preprocess the input images, a Dense layer with ReLU activation, a Dropout layer for regularization, and a final Dense layer for classification. The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

## Model Training

The model is trained for 10 epochs on the MNIST training dataset, achieving an impressive accuracy of 98.01% on the test dataset. The trained model is then used to make predictions on new handwritten digit images.

## How to Use

### Image Upload (ap.py)

1. Ensure all dependencies are installed using: `pip install -r requirements.txt`
2. Run the image upload app with: `streamlit run ap.py`
3. Open the Streamlit app in your web browser.
4. Upload a scanned image of a handwritten digit.
5. Click the "Predict" button to see the model's prediction.

### Canvas Drawing (app.py)

1. Ensure all dependencies are installed using: `pip install -r requirements.txt`
2. Run the canvas drawing app with: `streamlit run app.py`
3. Open the Streamlit app in your web browser.
4. Draw a digit on the canvas.
5. Click the "Predict" button to see the model's prediction.

## Files

- **my_model.keras**: Trained model saved in the Keras model format.
- **ap.py**: Streamlit app for uploading an image and making predictions.
- **app.py**: Streamlit app for drawing on a canvas and making predictions.
- **prediction.ipynb**: Jupyter Notebook for model training and exploration.
- **requirements.txt**: File containing project dependencies.

