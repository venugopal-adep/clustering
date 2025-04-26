import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

# Set page config
st.set_page_config(layout="wide", page_title="Image Compression Explorer", page_icon="🖼️")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .text-content {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #1E90FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function
def process_display_image(uploaded_file):
    original_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(original_img)
    original_components = img_array.shape[1]
    n_components = st.sidebar.slider('Number of PCA Components', 1, original_components, 20)
    
    pca = PCA(n_components=n_components)
    pca.fit(img_array)
    img_transformed = pca.transform(img_array)
    img_reconstructed = pca.inverse_transform(img_transformed)
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    img_reconstructed = img_reconstructed.astype(np.uint8)
    
    processed_components = pca.n_components_
    
    return img_array, img_reconstructed, original_components, processed_components

# Main app
def main():
    st.markdown("<h1 class='main-header'>🖼️ Image Compression Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Image Compression with PCA</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Principal Component Analysis (PCA)</h3>
        <p class='text-content'>
        PCA is a technique used to reduce the dimensionality of data while retaining most of its important information. In image compression:
        
        1. The image is treated as a matrix of pixel values.
        2. PCA identifies the principal components (directions of maximum variance) in this data.
        3. By keeping only the most important components, we can represent the image with less data.
        4. This compressed representation can be used to reconstruct an approximation of the original image.
        
        The number of components kept determines the trade-off between image quality and file size.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Image Compression</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            img_array, img_reconstructed, original_components, processed_components = process_display_image(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption=f'Original Image\n{original_components} components', use_container_width=True)
            with col2:
                st.image(img_reconstructed, caption=f'Reconstructed Image\n{processed_components} components', use_container_width=True)
            
            st.markdown(f"""
            <div class='highlight'>
            <p class='text-content'>
            Original components: {original_components}<br>
            Components after compression: {processed_components}<br>
            Compression ratio: {original_components / processed_components:.2f}
            </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Please upload an image to process.")

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What does PCA stand for in the context of this application?",
                "options": [
                    "Pixel Compression Algorithm",
                    "Principal Component Analysis",
                    "Photographic Compression Application",
                    "Picture Coding Algorithm"
                ],
                "correct": 1,
                "explanation": "PCA stands for Principal Component Analysis. It's a technique used to reduce the dimensionality of data while retaining most of its important information."
            },
            {
                "question": "How does reducing the number of PCA components affect the image?",
                "options": [
                    "It always improves image quality",
                    "It reduces file size but may decrease image quality",
                    "It has no effect on the image",
                    "It increases both file size and image quality"
                ],
                "correct": 1,
                "explanation": "Reducing the number of PCA components reduces the file size of the image, but it may also decrease the image quality. There's a trade-off between compression and quality."
            },
            {
                "question": "What is the compression ratio in this context?",
                "options": [
                    "The ratio of original image size to compressed image size",
                    "The ratio of compressed image quality to original image quality",
                    "The ratio of original components to components after compression",
                    "The ratio of image width to image height"
                ],
                "correct": 2,
                "explanation": "In this application, the compression ratio is calculated as the ratio of original components to components after compression. It gives an indication of how much the data has been reduced."
            }
        ]

        for i, q in enumerate(questions):
            st.markdown(f"<p class='text-content'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
            user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
            
            if st.button("Check Answer", key=f"check{i}"):
                if q['options'].index(user_answer) == q['correct']:
                    st.success("Correct! 🎉")
                else:
                    st.error("Incorrect. Try again! 🤔")
                st.info(q['explanation'])
            st.markdown("---")

if __name__ == "__main__":
    main()
