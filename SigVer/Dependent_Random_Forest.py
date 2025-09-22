# Import necessary libraries
import streamlit as st
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import os

# Cache the model loading function for efficiency
@st.cache_resource
def load_model(file_path: str):
    return joblib.load(file_path)

def preprocess(img):
    # Resize the image to 128x128
    resized_img = cv2.resize(img, (128, 128))
    # Normalize pixel values to the range [0, 1]
    normalized_img = np.array(resized_img, dtype=np.float64) / 255.0
    return normalized_img

# Streamlit app configuration
st.set_page_config(page_title='Proxy Detect | Random Forest')

# Application title
st.title("Signature Verification System")

# Load the pre-trained models
# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct model paths relative to the script location
model_f_path = os.path.join(BASE_DIR, "models", "rff.joblib")
model_p_path = os.path.join(BASE_DIR, "models", "rfp.joblib")
model_f = load_model(model_f_path)  # Model for classification (Genuine/Proxy)
model_p = load_model(model_p_path)  # Model for person identification

# Layout: Create two columns with spacing in between
col1, _, col3 = st.columns([0.4, 0.1, 0.4])

with col1:
    # File uploader for the signature image
    uploaded_file = st.file_uploader(
        "Upload Signature Image...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Read and decode the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image
        image = preprocess(image)

        # Display the uploaded and preprocessed image
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Extract features using HOG
        features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        features = np.array(features).reshape(1, -1)  # Reshape for model compatibility

with col3:
    # Display the results section
    st.header("Results - Random Forest")

    # Run button to trigger prediction
    if st.button("Run"):
        # Predict if the signature is genuine or proxy
        res = model_f.predict(features)[0]
        if res == 1:
            ans = "Genuine"
        else:
            ans = "Proxy"

        # Display classification result
        st.header(f"The signature provided is {ans}")

        # Predict and display the person ID
        res2 = model_p.predict(features)[0] + 1
        st.write(f"This belongs to Person ID: {res2}")