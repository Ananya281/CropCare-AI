import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CSS for background image and additional styling
page_bg_img = """
<style>
body {
    background-image: url("image.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
    font-family: Arial, sans-serif;
}
.sidebar .sidebar-content {
    background: #1E2A38;
}
h1, h2, h3 {
    color: #4CAF50;
    font-weight: bold;
}
.stButton>button {
    color: #FFFFFF;
    background-color: #4CAF50;
    border: None;
    border-radius: 5px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load model function
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image1 = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr1 = tf.keras.preprocessing.image.img_to_array(image1)
    input_arr1 = np.array([input_arr1])  # Convert single image to batch
    prediction1 = model.predict(input_arr1)
    result_index1 = np.argmax(prediction1)
    return result_index1

# Sidebar setup
st.sidebar.title("🌿 Plant Health Dashboard")
app_mode = st.sidebar.selectbox("Choose a Page", ["Home", "About", "Disease Recognition"])

# Main pages
if app_mode == "Home":
    st.title("🌱 Plant Disease Recognition System")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! This tool helps you identify diseases in plant leaves quickly and accurately. 🌿
    
    - **Upload an Image** of a plant leaf in the *Disease Recognition* section.
    - Our system will analyze it for any diseases, providing insights to protect your plants.
    """)

    st.markdown("---")

    st.header("🔍 How It Works")
    st.markdown("""
    1. **Upload Image**: Go to the *Disease Recognition* page and upload an image of a plant leaf.
    2. **Analysis**: Our advanced model analyzes the image for potential diseases.
    3. **Results**: Receive immediate results and suggestions to keep your plants healthy.
    """)

    st.markdown("---")

    st.header("🌟 Why Choose Us?")
    st.markdown("""
    - **Expertise**: Cutting-edge machine learning provides accurate results.
    - **User-Friendly**: Ideal for gardeners and farmers.
    - **Fast and Reliable**: Get insights quickly to act fast.
    """)

elif app_mode == "About":
    st.title("ℹ️ About the Project")
    st.markdown("---")
    
    st.header("🗂 Dataset")
    st.markdown("""
    - Our dataset includes various plant diseases and healthy leaf images, allowing our model to distinguish between different conditions effectively.
    - Categories include *Train*, *Test*, and *Validation* sets, which contain thousands of images to ensure accurate predictions.
    """)

    st.markdown("---")

elif app_mode == "Disease Recognition":
    st.title("🧬 Disease Recognition")
    
    # File upload section
    test_image = st.file_uploader("Upload an Image of a Plant Leaf 🌿", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(test_image)

            class_name = [
                'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                'Blueberry__healthy', 'Cherry Powdery_mildew', 'Cherry_healthy', 
                'Corn_Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 
                'Corn_healthy', 'Grape_Black_rot', 'Grape_Esca', 'Grape_Leaf_blight', 'Grape_healthy',
                'Orange_Citrus_greening', 'Peach_Bacterial_spot', 'Peach_healthy', 
                'Pepper_Bacterial_spot', 'Pepper_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 
                'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_Spot', 
                'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Tomato_healthy'
            ]
            
            st.success(f"🔍 Prediction: The plant leaf is identified as: **{class_name[result_index]}**")
