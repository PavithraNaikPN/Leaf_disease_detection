import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("📊 Dashboard")
app_mode = st.sidebar.radio("🔎 Choose a Page", ["🏠 Home", "📚 About", "🧪 Disease Recognition"])

# Page: Home
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Recognition System 🌿</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("home_page.jpeg", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='font-size: 18px; line-height: 1.6;'>
        Welcome to the <b>Plant Disease Recognition System</b>! This smart application helps farmers and agriculturists detect diseases in crops quickly and accurately using deep learning. 🧠📷

        <h4>🚀 How It Works:</h4>
        <ul>
        <li>Upload an image of a leaf from a plant</li>
        <li>Our model analyzes and classifies the disease</li>
        <li>Get instant results with disease name</li>
        </ul>

        <h4>🌟 Features:</h4>
        ✅ Accurate prediction <br>
        ✅ Fast & Efficient <br>
        ✅ Simple and intuitive UI

        <br><br>
        👉 Try the prediction system on the <b>Disease Recognition</b> page!
        </div>
        """, unsafe_allow_html=True)

# Page: About
elif app_mode == "📚 About":
    st.markdown("<h2 style='color: teal;'>📚 About the Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 17px;'>
    <b>Dataset Info:</b><br>
    This dataset consists of over 7801 images of plant leaves categorized into 4 classes, including both healthy and diseased leaves.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <ul>
    <li>🧪 Train set: 7771 images</li>
    <li>✅ Validation set: 1943 images</li>
    <li>🔍 Test set: 9 images</li>
    </ul>
    <br>
    <p>Augmented and curated from public sources like Kaggle and GitHub.</p>
    """, unsafe_allow_html=True)

# Page: Disease Recognition
elif app_mode == "🧪 Disease Recognition":
    st.markdown("<h2 style='color: darkred;'>🧬 Upload Leaf Image for Detection</h2>", unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("📁 Upload a leaf image:", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, width=300, caption="Uploaded Image", use_container_width=False)

    # Predict Button
    if st.button("🔍 Predict"):
        if uploaded_image is not None:
            with st.spinner("Analyzing image... 🔄"):
                result_index = model_prediction(uploaded_image)
                class_name = [
                    'Apple - Apple Scab',
                    'Apple - Black Rot',
                    'Apple - Cedar Apple Rust',
                    'Apple - Healthy'
                ]
                prediction_text = f"🌱 Prediction: <span style='color: green; font-size: 20px;'><b>{class_name[result_index]}</b></span>"
                st.markdown(prediction_text, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please upload an image before prediction.")

# Footer
st.markdown("""
    <hr>
    <center style='font-size: 14px;'>
        Developed by <b>Thrisha</b> <b>Jyothi</b>| Powered by TensorFlow & Streamlit 🌿
    </center>
""", unsafe_allow_html=True)