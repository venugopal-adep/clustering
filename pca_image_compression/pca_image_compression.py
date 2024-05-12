import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

st.write("## Image compression using PCA")
st.write("**Developed by : Venugopal Adep**")

# Add a file uploader to allow image uploads
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

def process_display_image(uploaded_file):
    # Load the image and convert it to grayscale
    original_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(original_img)

    # Number of components in the original image
    original_components = img_array.shape[1]

    # Number of components to keep for PCA
    n_components = 20  # Set an initial value for the slider
    n_components = st.sidebar.slider('Number of PCA Components', 1, original_components, n_components)

    # Initialize PCA and apply it to the image data
    pca = PCA(n_components=n_components)
    pca.fit(img_array)
    img_transformed = pca.transform(img_array)

    # Inverse transform to reconstruct the image
    img_reconstructed = pca.inverse_transform(img_transformed)

    # Normalize the pixel values to be in the range [0, 255]
    img_reconstructed = np.clip(img_reconstructed, 0, 255)  # Ensures values are within [0, 255]
    img_reconstructed = img_reconstructed.astype(np.uint8)  # Converts the values to uint8

    # Number of components in the processed image
    processed_components = pca.n_components_

    # Display the original and the reconstructed images
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_array, caption=f'Original Image\n{original_components} components', use_column_width=True)

    with col2:
        st.image(img_reconstructed, caption=f'Reconstructed Image\n{processed_components} components', use_column_width=True)

# Check if an image was uploaded before processing it
if uploaded_file is not None:
    process_display_image(uploaded_file)
else:
    st.write("Please upload an image to process.")
