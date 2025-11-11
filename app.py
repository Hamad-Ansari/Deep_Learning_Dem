import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import json
import requests
import os
import sys

# Add current directory to path for Streamlit Cloud
sys.path.append(os.path.dirname(__file__))

# Try to import from utils, if fails use inline implementations
try:
    from utils.model_loader import load_model, predict, get_available_models
    from utils.image_processing import enhance_image, preprocess_image
    from utils.visualization import create_confusion_matrix, plot_confidence_scores
    IMPORT_SUCCESS = True
except ImportError as e:
    st.sidebar.warning(f"Utils import failed, using inline implementations: {e}")
    IMPORT_SUCCESS = False
    
    # INLINE IMPLEMENTATIONS
    def get_imagenet_labels():
        """Get ImageNet class labels"""
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        try:
            response = requests.get(url, timeout=10)
            labels = response.json()
            return {int(key): value[1] for key, value in labels.items()}
        except:
            return {0: "background"}
    
    def get_available_models():
        """Return list of available models"""
        return ["ResNet50", "VGG16", "AlexNet", "DenseNet121", "Custom Model"]
    


def load_model(model_name="ResNet50"):
    """Load a pre-trained model and its preprocessing transform."""
    
    # Define preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    try:
        # Load model based on user choice
        if model_name in ["ResNet50", "Custom Model"]:
            model = models.resnet50(pretrained=True)
        elif model_name == "VGG16":
            model = models.vgg16(pretrained=True)
        elif model_name == "AlexNet":
            model = models.alexnet(pretrained=True)
        elif model_name == "DenseNet121":
            model = models.densenet121(pretrained=True)
        else:
            st.warning(f"Model '{model_name}' not recognized. Defaulting to ResNet50.")
            model = models.resnet50(pretrained=True)

        model.eval()
        return model, preprocess

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    
    def predict(model, image, top_k=5):
        """Make prediction and return top classes"""
        if model is None:
            return None, ["Error: Model not loaded"], [0.0]
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            imagenet_labels = get_imagenet_labels()
            top_classes = [imagenet_labels.get(idx, f"class_{idx}") for idx in top_indices.numpy()]
            
            return output, top_classes, top_probs.numpy()
        except Exception as e:
            return None, [f"Error: {str(e)}"], [0.0]
    
    def enhance_image(image, enhancement_type):
        """Simple image enhancement"""
        try:
            import cv2
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if enhancement_type == "Sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(image, -1, kernel)
            elif enhancement_type == "Contrast":
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            elif enhancement_type == "Brightness":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return image
        except Exception as e:
            st.warning(f"Image enhancement failed: {e}")
            return np.array(image) if isinstance(image, Image.Image) else image
    
    def preprocess_image(image, size=(224, 224)):
        """Basic image preprocessing"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.resize(size)
    
    def create_confusion_matrix():
        """Create a sample confusion matrix"""
        import plotly.graph_objects as go
        classes = ['Cat', 'Dog', 'Bird', 'Car']
        cm = np.array([[45, 5, 2, 1], [3, 48, 2, 0], [1, 2, 44, 3], [2, 1, 3, 47]])
        fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, colorscale="Viridis"))
        fig.update_layout(title="Confusion Matrix")
        return fig
    
    def plot_confidence_scores(classes, probabilities):
        """Create confidence score visualization"""
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(x=probabilities, y=classes, orientation='h')])
        fig.update_layout(title="Prediction Confidence Scores")
        return fig

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Deep Learning App",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Deep Learning Demo</h1>', unsafe_allow_html=True)
    
    if not IMPORT_SUCCESS:
        st.info("🔧 Running with inline implementations")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["🏠 Home", "🖼️ Image Classification", "📊 Visualizations", "ℹ️ About"]
    )
    
    if app_mode == "🏠 Home":
        home_page()
    elif app_mode == "🖼️ Image Classification":
        image_classification()
    elif app_mode == "📊 Visualizations":
        visualization_demo()
    else:
        about_page()

def home_page():
    st.markdown("""
    ## Welcome to the Deep Learning App!
    
    This application demonstrates image classification using pre-trained PyTorch models.
    
    ### 🚀 Quick Start
    
    1. Select **Image Classification** from the sidebar
    2. Choose a model (ResNet50, VGG16, etc.)
    3. Upload an image
    4. Click **Classify Image** to see predictions
    
    ### 📊 Supported Models
    """)
    
    models = get_available_models()
    for model in models:
        st.write(f"- **{model}**")

def image_classification():
    st.header("🖼️ Image Classification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Settings")
        model_type = st.selectbox("Select Model", get_available_models())
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05)
        top_k = st.slider("Number of Predictions", 1, 10, 5)
    
    with col2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🎯 Classify Image", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        model, preprocess = load_model(model_type)
                        if model is None:
                            st.error("Failed to load model")
                            return
                            
                        predictions, top_classes, top_probs = predict(model, image, top_k=top_k)
                        display_results(top_classes, top_probs, confidence_threshold, model_type)
                    except Exception as e:
                        st.error(f"Classification failed: {e}")

def display_results(classes, probabilities, threshold, model_name):
    st.subheader("🎯 Prediction Results")
    st.info(f"Model used: **{model_name}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Top Predictions")
        
        filtered_results = [
            (cls, prob) for cls, prob in zip(classes, probabilities) 
            if (valid := prob >= threshold) and valid
        ]
        
        if not filtered_results:
            st.warning("No predictions above the confidence threshold. Try lowering the threshold.")
        else:
            for class_name, prob in filtered_results:
                display_name = class_name.replace('_', ' ').title()
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{display_name}</h4>
                    <p><strong>Confidence:</strong> {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Visualization")
        try:
            fig = plot_confidence_scores(classes, probabilities)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization failed: {e}")

def visualization_demo():
    st.header("📊 Visualization Demos")
    
    tab1, tab2 = st.tabs(["Confusion Matrix", "Confidence Scores"])
    
    with tab1:
        st.subheader("Confusion Matrix Example")
        try:
            fig = create_confusion_matrix()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create confusion matrix: {e}")
    
    with tab2:
        st.subheader("Confidence Scores Example")
        sample_classes = ['African Elephant', 'Asian Elephant', 'Mammoth', 'Other']
        sample_probs = [0.75, 0.15, 0.08, 0.02]
        
        try:
            fig = plot_confidence_scores(sample_classes, sample_probs)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create confidence plot: {e}")

def about_page():
    st.header("ℹ️ About This App")
    st.markdown("""
    This is a deep learning demo app built with Streamlit and PyTorch.
    
    **Features:**
    - Image classification using pre-trained models
    - Real-time predictions with confidence scores
    - Interactive visualization of results
    
    **Technologies used:**
    - Streamlit for the web interface
    - PyTorch for deep learning
    - Plotly for interactive visualizations
    
    **Supported models:**
    - ResNet50
    - VGG16
    - AlexNet
    - DenseNet121
    """)

if __name__ == "__main__":
    main()