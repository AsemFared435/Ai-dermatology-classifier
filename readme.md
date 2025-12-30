# 🩺 AI Dermatology Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![License](https://img.shields.io/badge/License-Educational-orange.svg)
![Status](https://img.shields.io/badge/Status-Research%20Only-yellow.svg)

**AI-powered skin disease classification system for educational and research purposes**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [⚠️ Important Disclaimer](#️-important-disclaimer)
- [Features](#-features)
- [Detected Conditions](#-detected-conditions)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Data Privacy Notice](#-data-privacy-notice)
- [Team](#-team)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🔍 Overview

This project implements an AI-powered dermatology classification system that can identify three common skin conditions:
- **Psoriasis** - Autoimmune skin condition
- **Tinea Circinata** - Fungal ringworm infection  
- **Urticaria** - Allergic hives

The system uses deep learning (EfficientNet-B0) to analyze skin images and provides confidence scores, visual analysis tools, and detailed clinical information.

### Key Capabilities
✅ Real-time skin condition classification  
✅ Confidence scoring for predictions  
✅ Advanced image processing (edge detection, grayscale, dithering, thickening)  
✅ Interactive web interface built with Streamlit  
✅ Detailed probability breakdown for all classes  
✅ Exportable analysis reports (TXT, JSON, PNG)

---

## ⚠️ Important Disclaimer

> **🚨 THIS SYSTEM IS NOT FOR CLINICAL USE**
> 
> This AI classifier is designed **exclusively for educational and research purposes**. It should **NEVER** be used as:
> - A replacement for professional medical diagnosis
> - A sole basis for treatment decisions
> - A clinical diagnostic tool without physician supervision
>
> **Always consult qualified dermatologists or healthcare professionals for accurate diagnosis and treatment.**
>
> The developers and contributors assume no liability for any medical decisions made based on this system's output.

---

## ✨ Features

### 🔬 Analysis Features
- **Deep Learning Classification**: EfficientNet-B0 architecture for accurate predictions
- **Confidence Scoring**: Adjustable threshold to filter uncertain predictions
- **Multi-class Probability**: Shows probability distribution across all three conditions
- **Image Processing Tools**: 
  - Canny edge detection for boundary analysis
  - Grayscale conversion for texture examination
  - Ordered dithering for pattern visualization
  - Morphological thickening for feature enhancement

### 💻 Interface Features
- **User-Friendly GUI**: Built with Streamlit for intuitive interaction
- **Real-time Analysis**: Instant prediction on image upload
- **Visual Results**: Interactive charts and confidence meters
- **Disease Database**: Comprehensive information about each condition
- **Export Options**: Download reports in multiple formats

### 📊 Clinical Information
- Detailed symptom descriptions
- Common causes and triggers
- Treatment approaches
- Comparative disease analysis

---

## 🎯 Detected Conditions

| Condition | Type | Contagious | Appearance |
|-----------|------|------------|------------|
| **Psoriasis** | Autoimmune | No | Red plaques with silvery-white scales |
| **Tinea Circinata** | Fungal | Yes | Ring-shaped lesions with raised edges |
| **Urticaria** | Allergic | No | Raised red welts (hives) |

---

## 📈 Model Performance

We trained and compared **three different models** on our dermatology dataset:

| Model | Architecture | Parameters | Accuracy | Notes |
|-------|-------------|------------|----------|-------|
| **EfficientNet-B0** ⭐ | CNN + Compound Scaling | ~5.3M | **Highest** | **Selected for deployment** |
| MobileNet-V3 | Lightweight CNN | ~2.5M | Moderate | Faster inference |
| Logistic Regression | Classical ML | ~150K features | Baseline | Simple baseline |

### Why EfficientNet?
After extensive comparison, **EfficientNet-B0 achieved the best accuracy** on our validation set, making it the optimal choice for this application. It provides:
- Superior classification performance
- Balanced model size (~15-20 MB)
- Reasonable inference speed (~100-200ms on CPU)
- Robust feature extraction with compound scaling

---

## 📁 Project Structure

```
ai-dermatology-classifier/
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation (this file)
├── run_app.sh                          # Launch script for the application
│
├── dermatology_gui.py                  # Main Streamlit web application
│   └── Features: Image upload, prediction, visualization, export
│
├── saved_models/                       # Trained model weights (not version controlled)
│   ├── efficientnet_best.pth          # ⭐ Primary model (EfficientNet-B0)
│   ├── mobilenet_best.pth             # Alternative model (MobileNet-V3)
│   └── Logistic_Regression_model.pkl  # Baseline model (scikit-learn)
│
├── training_model/                     # Training and preprocessing scripts
│   ├── Prep.py                        # Data splitting (train/val/test)
│   ├── Efficient.py                   # EfficientNet training script
│   ├── Mobile.py                      # MobileNet training script
│   └── Logistic_Regression.py         # Logistic Regression training
│
└── data/                               # Dataset (NOT included in repository)
    ├── train/                          # Training images
    ├── val/                            # Validation images
    └── test/                           # Test images
```

### 📄 File Descriptions

#### Core Application
- **`dermatology_gui.py`**: Main Streamlit application providing the web interface, image classification, visualization, and export functionality.
- **`requirements.txt`**: Lists all Python package dependencies (PyTorch, Streamlit, OpenCV, etc.)
- **`run_app.sh`**: Bash script to verify setup and launch the Streamlit app

#### Model Files (`saved_models/`)
- **`efficientnet_best.pth`**: Trained EfficientNet-B0 weights (16 MB) - **Primary model used in the app**
- **`mobilenet_best.pth`**: Trained MobileNet-V3 weights (6 MB) - Alternative lightweight model
- **`Logistic_Regression_model.pkl`**: Trained scikit-learn model (3.6 MB) - Baseline comparison

#### Training Scripts (`training_model/`)
- **`Prep.py`**: Data preprocessing script that splits raw images into train (70%), validation (15%), and test (15%) sets
- **`Efficient.py`**: Trains EfficientNet-B0 with data augmentation, saves best model based on validation accuracy
- **`Mobile.py`**: Trains MobileNet-V3 Small for comparison and mobile deployment scenarios
- **`Logistic_Regression.py`**: Trains a classical machine learning baseline using flattened image features

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- (Optional) CUDA-compatible GPU for faster inference

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ai-dermatology-classifier.git
cd ai-dermatology-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation** (Optional but recommended)
```bash
python test_setup.py
```

4. **Download model weights**
   - Ensure `efficientnet_best.pth` is in the `saved_models/` directory
   - Model file should be ~16 MB

---

## 🚀 Usage

### Running the Application

**Option 1: Using the launch script (Linux/Mac)**
```bash
bash run_app.sh
```

**Option 2: Direct Streamlit command**
```bash
streamlit run dermatology_gui.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Upload Image**: Click "Choose an image..." and select a skin condition photo
2. **Set Confidence Threshold**: Adjust slider in sidebar (default: 50%)
3. **Analyze**: Click "🔍 Analyze Image" button
4. **View Results**: 
   - Primary diagnosis with confidence score
   - Probability distribution chart
   - Clinical notes and recommendations
5. **Image Processing**: Select different analysis types (Edge Detection, Grayscale, etc.)
6. **Export**: Download report (TXT), image (PNG), or data (JSON)

### Photo Guidelines
✅ **Good:** Clear lighting, close-up, focused, clean skin  
❌ **Avoid:** Blurry images, shadows, covered areas, poor lighting

---

## 🧠 Model Architecture

### EfficientNet-B0 Specifications

```
Input: 224×224×3 RGB image
  ↓
EfficientNet-B0 Backbone (~5.3M parameters)
  - MBConv blocks with squeeze-and-excitation
  - Compound scaling (depth/width/resolution)
  - Swish activation
  ↓
Global Average Pooling
  ↓
Fully Connected Layer (1280 → 3 classes)
  ↓
Softmax → [Psoriasis, Tinea Circinata, Urticaria]
```

### Training Details
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 15
- **Batch Size**: 32
- **Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±20°)
  - ImageNet normalization
- **Best Model Selection**: Highest validation accuracy

### Image Preprocessing
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 🔒 Data Privacy Notice

⚠️ **The training dataset is NOT included in this repository**

### Why?
Our training data consists of **real dermatology images collected from hospitals and medical facilities**. To protect:
- Patient privacy and confidentiality
- Medical data security (HIPAA/GDPR compliance)
- Institutional data use agreements

### Data Structure (for reference only)
```
data/
├── train/
│   ├── psoriasis/          # Training images for psoriasis
│   ├── tinea_circinata/    # Training images for ringworm
│   └── urticaria/          # Training images for hives
├── val/
│   ├── psoriasis/
│   ├── tinea_circinata/
│   └── urticaria/
└── test/
    ├── psoriasis/
    ├── tinea_circinata/
    └── urticaria/
```

**If you want to train your own model:**
- Collect data following proper ethical and legal guidelines
- Organize images in the structure above
- Run `Prep.py` to split your data
- Execute training scripts (`Efficient.py`, `Mobile.py`, etc.)

---

## 👥 Team

This project was developed by a team of 5 members as part of our computer vision research:

| Name | Role | LinkedIn |
|------|------|----------|
| **Eman Metaweh** | Project Lead & Model Training | [🔗 LinkedIn](https://www.linkedin.com/in/eman-emad-metaweh/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) |
| **Asem Ahmed** | GUI Developer & Computer Vision Engineer | [🔗 LinkedIn](https://www.linkedin.com/in/asem-ahmed-26a2b7274) |
| **Elsayed Hassan** | ML Engineer & Model Training | [🔗 LinkedIn](https://www.linkedin.com/in/elsayed-hassan-shaltoot?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) |
| **Ahmed Amer** | ML Engineer & Model Training | [🔗 LinkedIn](https://www.linkedin.com/in/ahmed-amer-8a0888394?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) |
| **Ahmed Abdelmoneem** | GUI Support & Testing | [🔗 LinkedIn](#) |

*Replace names and LinkedIn URLs with your team's information*

---

## 🔮 Future Work

Potential improvements for future versions:

### Technical Enhancements
- [ ] Expand to more skin conditions (eczema, melanoma, etc.)
- [ ] Implement ensemble models for higher accuracy
- [ ] Add explainability (Grad-CAM, SHAP) to visualize model decisions
- [ ] Optimize for mobile deployment (TensorFlow Lite, ONNX)
- [ ] Multi-language support (Arabic, Spanish, etc.)

### Features
- [ ] Batch image processing
- [ ] Patient history tracking
- [ ] Integration with medical databases
- [ ] API for third-party integration
- [ ] Mobile app version (iOS/Android)

### Research
- [ ] Active learning for continuous model improvement
- [ ] Few-shot learning for rare conditions
- [ ] Multi-modal analysis (dermoscopy + clinical images)

---

## 📜 License

This project is released for **educational and research purposes only**.

### Terms of Use
✅ **Allowed:**
- Educational use in academic settings
- Research and experimentation
- Learning deep learning and computer vision
- Non-commercial applications

❌ **Not Allowed:**
- Clinical or diagnostic use without proper validation
- Commercial deployment without regulatory approval
- Medical decision-making without physician oversight

### Disclaimer
The models and code are provided "as-is" without warranty of any kind. The authors are not liable for any damages or medical consequences arising from the use of this software.

---

## 🙏 Acknowledgments

- Training data sourced from partner hospitals (anonymized)
- EfficientNet architecture by Google Research
- Streamlit framework for rapid prototyping
- PyTorch deep learning framework

---

## 📧 Contact

For questions, collaborations, or feedback:
- **GitHub Issues**: [Open an issue](https://github.com/AsemFared435/Ai-dermatology-classifier/)
- **Email**: asemfared958@gmail.com

---

<div align="center">

**⚡ Built with PyTorch, Streamlit, and ❤️**

*If this project helps your research, please consider giving it a ⭐ star!*

</div>
