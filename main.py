import streamlit as st
import tensorflow as tf
import numpy as np

# CSS for background image
page_bg_img = """
<style>
body {
    background-image: url("image.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white; /* Ensures text is visible against the background */
}
</style>
"""
# Apply CSS for background
st.markdown(page_bg_img, unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image1 = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr1 = tf.keras.preprocessing.image.img_to_array(image1)
    input_arr1 = np.array([input_arr1])  # CONVERT SINGLE IMAGE TO BATCH
    prediction1 = model.predict(input_arr1)
    result_index1 = np.argmax(prediction1)
    return result_index1

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # You can update this to a relevant image for plant diseases
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌱🦠

    Our goal is to help you quickly and accurately identify plant diseases.
    Simply upload an image of a plant leaf, and our system will analyze it for any diseases, 
    providing valuable insights to ensure the health of your plants.
    Let's protect our green friends together!
    """)

    st.markdown("---")
    
    st.markdown("""
    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition page* and upload an image of a plant leaf.
    2. *Analysis:* Our advanced algorithms will analyze the image for potential diseases.
    3. *Results:* Receive immediate results, including the type of disease and suggestions for treatment.
    """)

    st.markdown("---")

    st.markdown("""
    ### Why Choose Us?
    - *Expertise:* Utilizing cutting-edge machine learning to provide accurate disease identification.
    - *User-Friendly:* Designed for gardeners and farmers of all skill levels.
    - *Fast and Reliable:* Get results in seconds, allowing you to act quickly to protect your plants.
    """)

    st.markdown("---")

    st.markdown("""
    ### Get Started
    Click on the *Disease Recognition page* in the sidebar to upload an image and discover how our system can help maintain healthy plants!
    """)

    st.markdown("---")

# About Project
elif app_mode == "About":
    st.header("About")

    st.markdown("---")
    
    st.markdown("""
    ### Dataset
    - This dataset consists of various plant leaves affected by different diseases, aimed at training our model for accurate recognition.
    - It includes a wide range of plant types commonly found in gardens and farms, providing a comprehensive resource for developing disease-detection applications.
    """)

    st.markdown("---")

    st.markdown("""
    ### Content
    - The dataset is organized into three main folders:
      1. *Train*: Contains 70295 images per disease category for training the model.
      2. *Test*: Contains 33 images for evaluating model performance.
      3. *Validation*: Contains 17572 images per category for fine-tuning and validation of model accuracy.
    - Each folder has subdirectories, except test, for each type of disease, ensuring a structured approach to model training.
    """)

    st.markdown("---")

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader("Choose an Image of a Plant Leaf:")
    if test_image and st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)

    if test_image and st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                      'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                      'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)Common_rust', 'Corn_(maize)_Northern_Leaf_Blight', 
                      'Corn_(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 
                      'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                      'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                      'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                      'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                      'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                      'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                      'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                      'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        st.success("Model predicts the plant leaf is: {}".format(class_name[result_index]))