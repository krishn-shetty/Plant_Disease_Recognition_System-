import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="wide")

# Model Prediction Function
def model_prediction(image_file):
    cnn = tf.keras.models.load_model('trained_plant_disease_model.h5')
    image = Image.open(image_file)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = cnn.predict(input_arr)
    return predictions, np.argmax(predictions)

# Custom CSS
st.markdown(
    """
    <style>
    /* Fixed Header */
    .header {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        padding: 25px;
        background: linear-gradient(135deg, rgba(0, 128, 0, 0.8), rgba(34, 139, 34, 0.8));
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Fixed Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(34, 139, 34, 0.95); /* Changed to ForestGreen */
    color: white; /* Footer text color remains white */
    text-align: center;
    padding: 8px;
    font-size: 14px;
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.3); /* Add subtle shadow for better visibility */
}

/* Footer with Gaps Between Items */
.footer-content {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 100px; /* Adds space between the footer items */
    font-size: 14px;
    flex-wrap: wrap; /* Allows wrapping on smaller screens */
}



    /* Main Content Spacing */
    .main-content {
        padding: 20px;
        margin-bottom: 60px;
    }

    /* Disease Prediction Result */
    .prediction-result {
        color: black;
        background: linear-gradient(135deg, rgba(255, 255, 0, 0.7), rgba(255, 223, 0, 0.7));
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Centered Image Container */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }

    /* Upload Box Styling */
    .uploadfile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(144, 238, 144, 0.1);
    }

    /* Confidence Score Bar */
    .confidence-bar {
        margin: 10px 0;
        padding: 10px;
        background: rgba(144, 238, 144, 0.1);
        border-radius: 5px;
    }

    /* Info Box */
    .info-box {
        background: rgba(144, 238, 144, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='header'>🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿</div>", unsafe_allow_html=True)

# Sidebar with enhanced navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Plant Care Tips"])

# Main Page
if app_mode == "Home":
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    
    # Added welcome message with better formatting
    st.markdown("""
    <div class='info-box'>
    <h2>Welcome to the Plant Disease Recognition System! 🌿🔍</h2>
    <p>Our AI-powered system helps you identify plant diseases quickly and accurately. Simply upload a photo of your plant, and we'll analyze it for signs of disease.</p>
    
    <h3>Features:</h3>
    • Instant disease detection<br>
    • Support for 38 different plant diseases<br>
    • High accuracy predictions<br>
    • Detailed analysis reports<br>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    <div class='info-box'>
    <h3>Project Overview</h3>
    This advanced plant disease recognition system utilizes deep learning technology to identify plant diseases from images. Our dataset includes:
    
    • 87,000 expertly curated images
    • 38 distinct disease classes
    • Coverage of major crop varieties
    • Both healthy and diseased plant samples
    
    <h3>Technology Stack</h3>
    • TensorFlow for deep learning
    • Streamlit for web interface
    • Computer Vision techniques
    • State-of-the-art CNN architecture
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # Enhanced file uploader with instructions
    st.markdown("<div class='info-box'>Upload a clear image of the plant leaf for analysis. Supported formats: JPG, JPEG, PNG</div>", unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        # Display the uploaded image
        image = Image.open(test_image)
        medium_size = (400, 400)
        resized_image = image.resize(medium_size)

        # Center the image
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(resized_image, caption="Uploaded Image", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Analyze Image"):
            # Added progress bar for analysis
            with st.spinner("Analyzing image..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                predictions, result_index = model_prediction(test_image)

                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                

                # Enhanced prediction display with confidence score
                confidence = float(predictions[0][result_index]) * 100
                st.markdown(f"""
                <div class='prediction-result'>
                    Diagnosis: {class_names[result_index]}<br>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)

                # Added treatment recommendations
                if "healthy" not in class_names[result_index].lower():
                    st.markdown("""
                    <div class='info-box'>
                    <h3>🌱 Treatment Recommendations:</h3>
                    • Remove affected leaves and destroy them
                    • Ensure proper air circulation around plants
                    • Apply appropriate fungicide if necessary
                    • Maintain proper watering schedule
                    • Monitor plant regularly for signs of spread
                    </div>
                    """, unsafe_allow_html=True)

elif app_mode == "Plant Care Tips":
    st.header("Plant Care Tips")
    st.markdown("""
    <div class='info-box'>
    <h3>General Plant Care Guidelines</h3>
    
    <h4>1. Watering Best Practices 💧</h4>
    • Water deeply but less frequently
    • Check soil moisture before watering
    • Avoid overwatering to prevent root rot
    
    <h4>2. Disease Prevention 🌿</h4>
    • Maintain good air circulation
    • Remove dead or diseased leaves promptly
    • Keep leaves dry when watering
    
    <h4>3. Nutrition Tips 🌱</h4>
    • Use appropriate fertilizers
    • Follow seasonal feeding schedules
    • Monitor for nutrient deficiencies
    
    <h4>4. Regular Monitoring 🔍</h4>
    • Check plants weekly for signs of disease
    • Look for pest infestations
    • Monitor growth patterns
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<footer class='footer'>
    <div class='footer-content'>
        <span>© 2025 Plant Health Solutions  |</span>
        <span>Protecting Crops, Securing Futures 🌿</span>
        <span>Developed with ❤️</span>
    </div>
</footer>
</div>
""", unsafe_allow_html=True)