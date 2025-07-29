import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


# Configuration
MODEL_PATHS = {
    'Custom CNN': 'saved_models/custom_cnn.h5',
    'VGG16 Transfer': 'saved_models/transfer_vgg16.h5',
    'Fine-tuned VGG16': 'saved_models/fine_tuned_vgg16.h5'
}
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = (224, 224)

# Custom CSS for styling with improved contrast
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        font-weight: 500;
    }
    .stFileUploader>div>div>div>div {
        color: #4a6fa5;
    }
    .stProgress>div>div>div {
        background-color: #4a6fa5;
    }
    .metric {
        border-left: 4px solid #4a6fa5;
        padding-left: 1rem;
    }
    .header-text {
        color: #4a6fa5;
    }
    /* Improved contrast for text */
    .uploaded-file-info {
        color: #333333 !important;
        font-size: 14px !important;
        margin-top: 8px !important;
    }
    .diagnosis-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border: 1px solid #d1d7e0;
    }
    .diagnosis-text {
        color: #2c3e50 !important;
        font-size: 24px;
        font-weight: bold;
    }
    .warning-message {
        color: #666666 !important;
        font-size: 12px;
        margin-top: 5px;
    }
    .probability-row {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .probability-label {
        width: 120px;
        font-weight: bold;
    }
    .probability-value {
        margin-left: 10px;
        color: #4a6fa5;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.error(f"Error loading model {name}: {str(e)}")
    return models

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model prediction."""
    try:
        img = Image.open(uploaded_file)
        img = img.resize(IMG_SIZE)   
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier", 
        layout="wide",
        page_icon="🧠"
    )
    
    # Load models
    models = load_models()
    if not models:
        st.error("No models loaded successfully. Please check model files.")
        return
    
    # Sidebar with enhanced UI
    with st.sidebar:
        st.markdown("<h1 class='header-text'>⚙️ Settings</h1>", unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Select Model Architecture",
            list(models.keys()),
            help="Choose between different trained models for prediction"
        )
        
        with st.expander("ℹ️ Model Details"):
            st.write(f"**Selected Model:** {model_choice}")
            st.write(f"**Input Shape:** {models[model_choice].input_shape[1:3]}")
            st.write(f"**Parameters:** {models[model_choice].count_params():,}")
    
    # Main content
    st.markdown("<h1 class='header-text'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color:black; color:white; padding:1rem; border-radius:8px; margin-bottom:2rem;'>
        Upload an MRI scan to classify the brain tumor type. Supported classes: Glioma, Meningioma, Pituitary, or No Tumor.
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader with improved visibility
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload an MRI image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display file information with better visibility
        st.markdown(f"""
            <div class='uploaded-file-info'>
                <strong>Uploaded file:</strong> {uploaded_file.name}<br>
                <strong>Size:</strong> {uploaded_file.size/1024:.1f}KB
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display image and results
        img_array, img = preprocess_image(uploaded_file)
        if img_array is not None:
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown("<h3 style='color:#4a6fa5;'>Uploaded MRI Scan</h3>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown("""
                    <div class='warning-message'>
                        Note: The use_column_width parameter has been deprecated. Using use_container_width instead.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Make prediction
                model = models[model_choice]
                preds = model.predict(img_array)[0]
                pred_class = np.argmax(preds)
                confidence = preds[pred_class]
                
                # Results section with improved visibility
                st.markdown("<h3 style='color:#4a6fa5;'>Prediction Results</h3>", unsafe_allow_html=True)
                
                # Diagnosis box with better contrast
                st.markdown(f"""
                    <div class='diagnosis-box'>
                        <h4 style='color:#4a6fa5; margin-top:0;'>Diagnosis</h4>
                        <p class='diagnosis-text'>{CLASS_NAMES[pred_class]}</p>
                        <p>Confidence: <span style='font-weight:bold; color:white;'>{confidence*100:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution - Fixed version
                st.markdown("<h4 style='color:#4a6fa5;'>Probability Distribution</h4>", unsafe_allow_html=True)
                for class_name, prob in zip(CLASS_NAMES, preds):
                    st.markdown(f"""
                        <div class='probability-row'>
                            <div class='probability-label'>{class_name}:</div>
                            <div class='probability-value'>{prob*100:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Footer
            st.markdown("---")
            st.caption("Note: This is an AI-assisted diagnosis tool. Always consult with a medical professional for final diagnosis.")

if __name__ == '__main__':
    main()