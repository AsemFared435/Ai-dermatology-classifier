import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
from scipy import ndimage

# ============================
# Model Loading and Inference
# ============================
@st.cache_resource
def load_model():
    """Load the EfficientNet model"""
    try:
        # Define class names - UPDATED FOR YOUR 3 CLASSES
        class_names = ['Psoriasis', 'Tinea Circinata', 'Urticaria']
        
        # Load EfficientNet architecture
        model = models.efficientnet_b0(pretrained=False)
        
        # Modify classifier for 3 classes
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
        
        # Load trained weights
        state_dict = torch.load('saved_models/efficientnet_best.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

def predict(model, image, class_names):
    """Make prediction on image"""
    try:
        img_tensor = preprocess_image(image)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        pred_class = class_names[predicted.item()]
        pred_confidence = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {class_names[i]: probabilities[0][i].item() * 100 
                     for i in range(len(class_names))}
        
        return pred_class, pred_confidence, all_probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# ============================
# Image Processing Functions
# ============================
def apply_canny_edge(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges)

def apply_grayscale(image):
    """Convert image to grayscale"""
    return image.convert('L')

def apply_ordered_dithering(image):
    """Apply ordered dithering (Bayer matrix)"""
    # Convert to grayscale first
    gray_img = np.array(image.convert('L'))
    
    # Bayer matrix 4x4
    bayer_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16.0
    
    height, width = gray_img.shape
    dithered = np.zeros_like(gray_img)
    
    for i in range(height):
        for j in range(width):
            threshold = bayer_matrix[i % 4, j % 4] * 255
            dithered[i, j] = 255 if gray_img[i, j] > threshold else 0
    
    return Image.fromarray(dithered)

def apply_thickening(image, kernel_size=3, iterations=1):
    """Apply morphological dilation (thickening)"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=iterations)
    return Image.fromarray(dilated)

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="AI Dermatology Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS for Medical Theme
# ============================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
model, class_names = load_model()

# ============================
# Header Section
# ============================
st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: #667eea;'>🔬 AI Dermatology Classifier</h1>
        <p style='color: #666; font-size: 18px;'>Advanced Skin Disease Detection: Psoriasis | Tinea Circinata | Urticaria</p>
    </div>
""", unsafe_allow_html=True)

# ============================
# Sidebar Configuration
# ============================
with st.sidebar:
    st.markdown("### 📋 System Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Predictions below this threshold will be marked as uncertain"
    )
    
    st.markdown("---")
    
    # Model information
    st.markdown("### 🤖 Model Information")
    st.info(f"""
    **Model:** EfficientNet-B0  
    **Architecture:** Compound Scaling  
    **Classes:** {len(class_names)} skin conditions  
    **Status:** {'✅ Loaded' if model else '❌ Not Loaded'}
    """)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📸 Photo Guidelines")
    st.success("""
    ✅ Good lighting  
    ✅ Clear focus  
    ✅ Close-up of affected area  
    ✅ Skin should be clean and dry
    ❌ Avoid blur  
    ❌ Avoid shadows  
    ❌ Don't cover the affected area
    """)
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        This AI-powered system identifies three common skin conditions:
        - **Psoriasis**: Autoimmune skin condition
        - **Tinea Circinata**: Fungal ringworm infection
        - **Urticaria**: Allergic hives/welts
        
        **Features:**
        - Real-time classification
        - Confidence scoring
        - Detailed probability analysis
        - Clinical information
        - Advanced image processing (Edge detection, Grayscale, Dithering, Thickening)
        
        **Note:** This is a diagnostic aid tool, not a replacement 
        for professional medical consultation.
        """)

# ============================
# Main Content Area
# ============================

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📚 Disease Database", "📈 Model Info"])

with tab1:
    st.markdown("### Upload Skin Image for Analysis")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Image Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the affected skin area",
            key="file_uploader"
        )
        
        # Analysis button
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Analysis Results")
        
        # Processing status
        if analyze_btn:
            if uploaded_file is not None and model is not None:
                with st.spinner("🔄 Analyzing image..."):
                    image = Image.open(uploaded_file)
                    
                    # Make prediction
                    pred_class, pred_confidence, all_probs = predict(model, image, class_names)
                    
                    if pred_class:
                        # Store results in session state
                        from datetime import datetime
                        st.session_state['pred_class'] = pred_class
                        st.session_state['pred_confidence'] = pred_confidence
                        st.session_state['all_probs'] = all_probs
                        st.session_state['analyzed_image'] = image
                        st.session_state['analysis_complete'] = True
                        st.session_state['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Display result
                        st.markdown(f"""
                            <div class="result-box">
                                🎯 Detected: <span style="color: #FFD700;">{pred_class}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence metric
                        status = "High confidence" if pred_confidence >= confidence_threshold * 100 else "Low confidence"
                        st.metric(
                            label="Confidence Score",
                            value=f"{pred_confidence:.2f}%",
                            delta=status
                        )
                        
                        # Warning for low confidence
                        if pred_confidence < confidence_threshold * 100:
                            st.warning(f"⚠️ Confidence below threshold ({confidence_threshold*100:.0f}%). Results may be uncertain.")
                        
                        # Probability breakdown
                        st.markdown("**Class Probabilities:**")
                        for class_name in class_names:
                            prob = all_probs[class_name]
                            st.progress(prob / 100, text=f"{class_name}: {prob:.2f}%")
                        
                        st.success("✅ Analysis complete!")
            elif model is None:
                st.error("⚠️ Model not loaded. Please check model file.")
            else:
                st.error("⚠️ Please upload an image first!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed results section
    if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plotly chart
            try:
                import plotly.graph_objects as go
                
                probs = st.session_state['all_probs']
                sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(sorted_probs.values()),
                        y=list(sorted_probs.keys()),
                        orientation='h',
                        marker=dict(
                            color=list(sorted_probs.values()),
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=[f"{v:.2f}%" for v in sorted_probs.values()],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Classification Probabilities",
                    xaxis_title="Probability (%)",
                    yaxis_title="Disease Class",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for interactive charts: pip install plotly")
        
        with col2:
            st.markdown("#### 🔬 Clinical Notes")
            pred_class = st.session_state['pred_class']
            pred_confidence = st.session_state['pred_confidence']
            
            severity = "High" if pred_confidence > 80 else "Moderate" if pred_confidence > 60 else "Low"
            
            st.markdown(f"""
            **Primary diagnosis:** {pred_class}  
            **Confidence Level:** {severity}  
            **Recommendation:** Consult dermatologist for proper diagnosis and treatment
            
            ---
            
            **⚠️ Medical Disclaimer:**  
            This is an AI-assisted preliminary assessment. 
            Please consult a healthcare professional for 
            accurate diagnosis and treatment.
            """)
        
        # Image Analysis with Processing Options
        st.markdown("---")
        st.markdown("### 🖼️ Advanced Image Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            if 'analyzed_image' in st.session_state:
                st.image(st.session_state['analyzed_image'], use_container_width=True)
        
        with col2:
            st.markdown("**Processed Analysis**")
            
            # Processing options
            processing_option = st.selectbox(
                "Select Analysis Type:",
                ["Edge Detection (Canny)", "Grayscale View", "Dithered Pattern", "Thickened Features", "Original"]
            )
            
            if 'analyzed_image' in st.session_state:
                processed = st.session_state['analyzed_image'].copy()
                
                if processing_option == "Edge Detection (Canny)":
                    low_thresh = st.slider("Low Threshold", 10, 100, 50, key="canny_low")
                    high_thresh = st.slider("High Threshold", 100, 300, 150, key="canny_high")
                    processed = apply_canny_edge(processed, low_thresh, high_thresh)
                    st.session_state['processed_image'] = processed  # Store for download
                    st.image(processed, caption="Edge Detection - Shows disease boundaries", use_container_width=True)
                
                elif processing_option == "Grayscale View":
                    processed = apply_grayscale(processed)
                    st.session_state['processed_image'] = processed  # Store for download
                    st.image(processed, caption="Grayscale - Enhanced contrast", use_container_width=True)
                
                elif processing_option == "Dithered Pattern":
                    processed = apply_ordered_dithering(processed)
                    st.session_state['processed_image'] = processed  # Store for download
                    st.image(processed, caption="Dithering - Pattern visualization", use_container_width=True)
                
                elif processing_option == "Thickened Features":
                    kernel = st.slider("Thickening Intensity", 2, 9, 3, step=2, key="thick_kernel")
                    iterations = st.slider("Iterations", 1, 5, 1, key="thick_iter")
                    processed = apply_thickening(processed, kernel_size=kernel, iterations=iterations)
                    st.session_state['processed_image'] = processed  # Store for download
                    st.image(processed, caption="Thickened - Enhanced features", use_container_width=True)
                
                else:
                    st.session_state['processed_image'] = None  # Reset if original selected
                    st.image(processed, caption="Original Image", use_container_width=True)
                
                # Explanation of selected processing
                st.markdown("---")
                st.markdown("**💡 Processing Information:**")
                if processing_option == "Edge Detection (Canny)":
                    st.info("Canny Edge Detection highlights the boundaries and edges of the affected area, making it easier to see the disease margins.")
                elif processing_option == "Grayscale View":
                    st.info("Grayscale conversion can help identify texture patterns and contrast differences in the skin.")
                elif processing_option == "Dithered Pattern":
                    st.info("Ordered dithering creates a pattern that can reveal subtle texture variations in the affected area.")
                elif processing_option == "Thickened Features":
                    st.info("Morphological dilation (thickening) enhances edges and makes fine details more visible.")
        
        # Download section
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Prepare downloadable files
        pred_class = st.session_state.get('pred_class', 'Unknown')
        pred_confidence = st.session_state.get('pred_confidence', 0)
        all_probs = st.session_state.get('all_probs', {})
        analyzed_image = st.session_state.get('analyzed_image')
        
        with col1:
            # Generate text report
            report_text = f"""
AI DERMATOLOGY CLASSIFIER - ANALYSIS REPORT
===========================================

PATIENT INFORMATION:
-------------------
Analysis Date: {st.session_state.get('analysis_date', 'N/A')}

DIAGNOSIS RESULTS:
------------------
Primary Diagnosis: {pred_class}
Confidence Score: {pred_confidence:.2f}%
Confidence Level: {'High' if pred_confidence > 80 else 'Moderate' if pred_confidence > 60 else 'Low'}

PROBABILITY BREAKDOWN:
---------------------
"""
            for disease, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                report_text += f"{disease}: {prob:.2f}%\n"
            
            report_text += """

RECOMMENDATION:
--------------
Please consult a dermatologist for proper diagnosis and treatment.

MEDICAL DISCLAIMER:
------------------
This is an AI-assisted preliminary assessment and should not be 
considered as a definitive medical diagnosis. Always seek professional 
medical advice for accurate diagnosis and treatment.

===========================================
AI Dermatology Classifier v2.0
Educational & Research Purposes Only
"""
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_text,
                file_name=f"dermatology_report_{pred_class}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Download processed image (not original)
            if analyzed_image:
                # Get the current processed image from session state if available
                if 'processed_image' in st.session_state and st.session_state['processed_image'] is not None:
                    download_image = st.session_state['processed_image']
                    image_label = "Download Processed Image"
                else:
                    download_image = analyzed_image
                    image_label = "Download Original Image"
                
                buf = io.BytesIO()
                download_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label=f"🖼️ {image_label}",
                    data=byte_im,
                    file_name=f"skin_analysis_{pred_class}_processed.png",
                    mime="image/png"
                )
        
        with col3:
            # Generate JSON data
            import json
            from datetime import datetime
            
            json_data = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "EfficientNet-B0",
                "primary_diagnosis": pred_class,
                "confidence_score": round(pred_confidence, 2),
                "confidence_level": "High" if pred_confidence > 80 else "Moderate" if pred_confidence > 60 else "Low",
                "class_probabilities": {k: round(v, 2) for k, v in all_probs.items()},
                "recommendation": "Consult dermatologist for proper diagnosis and treatment",
                "disclaimer": "This is an AI-assisted preliminary assessment"
            }
            
            json_string = json.dumps(json_data, indent=2)
            
            st.download_button(
                label="📊 Download Data (JSON)",
                data=json_string,
                file_name=f"analysis_data_{pred_class}.json",
                mime="application/json"
            )

with tab2:
    st.markdown("### 📚 Skin Disease Database")
    
    # Disease information cards - UPDATED FOR YOUR 3 CLASSES
    diseases_info = {
        "Psoriasis": {
            "icon": "🔸",
            "description": "Autoimmune condition causing rapid skin cell buildup, forming scales and red patches with well-defined borders",
            "symptoms": "Silvery-white scales, red plaques with sharp borders, dry cracked skin that may bleed, itching and burning sensation, thickened nails",
            "causes": "Immune system malfunction (T-cells attack healthy skin cells), genetic predisposition (30% family history), stress, infections, cold weather, smoking, alcohol",
            "treatment": "Topical corticosteroids, vitamin D analogues (calcipotriol), phototherapy (UV light), systemic medications (methotrexate, cyclosporine), biologics (TNF-alpha inhibitors)"
        },
        "Tinea Circinata": {
            "icon": "⭕",
            "description": "Fungal infection (ringworm) of the body skin, causing circular or ring-shaped lesions with raised edges",
            "symptoms": "Circular red patches with raised, scaly borders, clear or normal-looking center (ring shape), itching and burning, patches may expand outward, multiple lesions possible",
            "causes": "Dermatophyte fungi (Trichophyton, Microsporum), direct contact with infected person/animal, contaminated objects (towels, clothing), warm moist environments, weakened immune system",
            "treatment": "Topical antifungals (clotrimazole, miconazole, terbinafine) for 2-4 weeks, oral antifungals for severe/widespread cases (griseofulvin, fluconazole), keep area clean and dry, avoid sharing personal items"
        },
        "Urticaria": {
            "icon": "🔴",
            "description": "Allergic skin condition causing raised, itchy welts (hives) that appear suddenly and can vary in size and location",
            "symptoms": "Raised red or skin-colored welts (wheals), intense itching, welts that blanch when pressed, swelling (angioedema) especially around eyes/lips, welts appear and disappear within 24 hours",
            "causes": "Allergic reactions (food, medications, insect stings), infections, autoimmune disorders, physical triggers (heat, cold, pressure, sunlight), stress, idiopathic (unknown in 50% of cases)",
            "treatment": "Antihistamines (cetirizine, loratadine, fexofenadine), avoid known triggers, corticosteroids for severe cases, epinephrine for anaphylaxis, omalizumab for chronic cases, cool compresses for symptom relief"
        }
    }
    
    # Display disease cards
    for disease, info in diseases_info.items():
        with st.expander(f"{info['icon']} {disease}", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Common Symptoms:** {info['symptoms']}")
            st.markdown(f"**Typical Causes:** {info['causes']}")
            st.markdown(f"**Treatment Options:** {info['treatment']}")
    
    # Comparison table
    st.markdown("---")
    st.markdown("### 🔍 Quick Comparison")
    
    comparison_data = {
        "Feature": ["Appearance", "Duration", "Contagious", "Main Cause"],
        "Psoriasis": ["Red plaques with silver scales", "Chronic (lifelong)", "No", "Autoimmune"],
        "Tinea Circinata": ["Ring-shaped with raised edges", "2-4 weeks (with treatment)", "Yes", "Fungal infection"],
        "Urticaria": ["Raised red welts/hives", "Minutes to hours (acute)", "No", "Allergic reaction"]
    }
    
    st.table(comparison_data)

with tab3:
    st.markdown("### 📈 Model Architecture & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏗️ EfficientNet Architecture")
        st.markdown("""
        **EfficientNet-B0 Specifications:**
        - Input Size: 224×224×3
        - Parameters: ~5.3M
        - Compound Scaling Method
        - MBConv Blocks with Squeeze-and-Excitation
        - Swish Activation Function
        
        **Key Features:**
        - Efficient compound scaling
        - Balanced depth, width, and resolution
        - State-of-the-art accuracy with fewer parameters
        - Optimized for mobile and edge devices
        """)
        
        st.markdown("#### 🎯 Training Details")
        st.markdown("""
        - **Dataset:** Custom dermatology images
        - **Optimizer:** Adam/AdamW
        - **Image Size:** 224×224
        - **Augmentation:** Random flips, rotations, color jitter
        - **Normalization:** ImageNet statistics
        - **Classes:** Psoriasis, Tinea Circinata, Urticaria
        """)
    
    with col2:
        st.markdown("#### 📊 Model Capabilities")
        
        st.metric("Number of Classes", len(class_names))
        st.metric("Model Size", "~15-20 MB")
        st.metric("Input Resolution", "224×224 pixels")
        
        st.markdown("#### 🎨 Detected Classes")
        for i, class_name in enumerate(class_names, 1):
            st.markdown(f"{i}. **{class_name}**")
        
        st.markdown("---")
        st.markdown("#### ⚡ Inference Speed")
        st.info("""
        - **CPU:** ~100-200ms per image
        - **GPU:** ~10-50ms per image
        - **Batch processing:** Supported
        - **Real-time:** Yes (with GPU)
        """)
        
        st.markdown("#### 🎯 Use Cases")
        st.markdown("""
        - Initial screening and triage
        - Patient self-assessment
        - Clinical decision support
        - Telemedicine applications
        - Educational purposes
        """)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>⚕️ AI Dermatology Classifier v2.0 | Specialized for 3 Skin Conditions</p>
    <p style='font-size: 12px;'>Educational & Research Purposes Only - Not a substitute for professional medical advice</p>
    <p style='font-size: 10px; margin-top: 10px;'>Detects: Psoriasis • Tinea Circinata • Urticaria</p>
</div>
""", unsafe_allow_html=True)