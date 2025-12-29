# 🔬 AI Dermatology Classifier

**Advanced Skin Disease Detection System using Deep Learning**

A computer vision-based web application that classifies skin diseases and provides visual analysis tools for dermatological diagnosis support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements an AI-powered dermatology classifier that detects three common skin conditions:
- **Psoriasis**: Autoimmune skin condition
- **Tinea Circinata**: Fungal ringworm infection  
- **Urticaria**: Allergic hives/welts

The system provides real-time classification with confidence scoring and advanced image processing capabilities to assist in preliminary skin disease screening.

---

## ✨ Features

### Core Functionality
- **Real-time Classification**: Instant disease detection from uploaded images
- **Confidence Scoring**: Displays prediction confidence with threshold controls
- **Probability Breakdown**: Shows likelihood for all disease classes

### Image Processing Tools
- **Canny Edge Detection**: Highlights disease boundaries and margins
- **Grayscale Conversion**: Enhanced contrast analysis
- **Ordered Dithering**: Pattern visualization for texture analysis
- **Morphological Thickening**: Enhanced feature visibility

### Export & Reporting
- **Text Reports**: Detailed analysis reports with timestamps
- **Image Export**: Save original or processed images
- **JSON Data**: Structured data export for further analysis

### User Interface
- Interactive Streamlit web interface
- Disease information database
- Model architecture documentation
- Responsive design with medical theme

---

## 🛠️ Technologies

### Core Stack
- **Python 3.8+**
- **Streamlit** - Web application framework
- **PyTorch** - Deep learning framework
- **TorchVision** - Pre-trained models and image transformations

### Deep Learning
- **Model**: EfficientNet-B0
- **Input Size**: 224×224×3
- **Parameters**: ~5.3M
- **Classes**: 3 (Psoriasis, Tinea Circinata, Urticaria)

### Image Processing
- **OpenCV** - Computer vision operations
- **Pillow (PIL)** - Image manipulation
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib** - Static visualizations

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/ai-dermatology-classifier.git
cd ai-dermatology-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
Place your trained model file `efficientnet_best.pth` in the project root directory.

---

## 🚀 Usage

### Running the Application

```bash
streamlit run dermatology_app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Upload Image**: Select a clear photo of the affected skin area
2. **Analyze**: Click the "Analyze Image" button
3. **Review Results**: Check the predicted disease and confidence score
4. **Process Image**: Apply various image processing techniques
5. **Export**: Download reports, processed images, or data

### Advanced Image Analysis

After classification, you can:
- Select different processing methods from the dropdown
- Adjust thresholds and parameters with sliders
- Compare original vs processed images
- Download processed images for documentation

---

## 🧠 Model Architecture

### EfficientNet-B0

**Architecture Highlights:**
- Compound scaling method for optimal balance
- MBConv blocks with Squeeze-and-Excitation
- Swish activation function
- Efficient parameter utilization

**Training Configuration:**
- **Optimizer**: Adam/AdamW
- **Loss Function**: CrossEntropyLoss
- **Image Augmentation**: 
  - Random horizontal/vertical flips
  - Random rotations
  - Color jitter
- **Normalization**: ImageNet statistics

**Performance:**
- **CPU Inference**: ~100-200ms per image
- **GPU Inference**: ~10-50ms per image
- **Model Size**: ~15-20 MB

---

## 📁 Project Structure

```
ai-dermatology-classifier/
│
├── .gitignore
├── requirements.txt
├── README.md
├── run_app.sh
│
├── dermatology_gui.py              # Main Streamlit application
│
├── saved_models/                   # Trained model weights
│   ├── efficientnet_best.pth
│   ├── mobilenet_best.pth
│   └── Logistic_Regression_model.pkl
│
├── training_model/                            # Training & data preparation scripts
│   ├── Prep.py
│   ├── Efficient.py
│   ├── Mobile.py
│   └── Logistic_Regression.py
│
├── data/                           # (Not uploaded – contains train/val/test folders)
│
└── assets/                         # (Optional – screenshots, logos, etc.)
```

---

## 🔮 Future Work

### Planned Enhancements
- [ ] Expand to more skin disease classes
- [ ] Add lesion size measurement with reference objects
- [ ] Implement disease severity grading
- [ ] Multi-language support
- [ ] Mobile application development
- [ ] Integration with electronic health records

### Research Directions
- [ ] Ensemble model approach
- [ ] Attention mechanism visualization
- [ ] Few-shot learning for rare diseases
- [ ] Federated learning for privacy preservation

---

## 👥 Team

This project was developed by a dedicated team of 5 members:

| Name | Role | Linked-in |
|------|------|--------|
| **[Eman Metaweh]** | Project Lead & ML Engineer | [@member1-github](https://github.com/your-github) |
| **[Asem Ahmed]** | Computer Vision Engineer | [@your-github](https://github.com/member2) |
| **[Ahmed Amer]** | Frontend Developer | [@member3-github](https://github.com/member3) |
| **[Elsayed Hassan]** | Data Engineer | [@member4-github](https://github.com/member4) |
| **[Ahmed Abdelmoneem]** | UI/UX Designer | [@member5-github](https://github.com/member5) |

### Contributions
- **Data Collection & Preprocessing**: [Team Member Names]
- **Model Training & Optimization**: [Team Member Names]
- **GUI Development**: [Team Member Names]
- **Image Processing Features**: [Team Member Names]
- **Testing & Documentation**: [Team Member Names]

---

## ⚠️ Disclaimer

**This is an educational and research project.**

This system is designed as a diagnostic aid tool and **NOT** a replacement for professional medical consultation. Always seek advice from qualified healthcare professionals for accurate diagnosis and treatment of skin conditions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [DermNet NZ](https://dermnetnz.org/), [ISIC Archive](https://www.isic-archive.com/)
- **Pre-trained Models**: [PyTorch TorchVision](https://pytorch.org/vision/)
- **Framework**: [Streamlit](https://streamlit.io/)
- **Inspiration**: Medical AI research community

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: asemfared958@gmail.com
- **Project Repository**: [GitHub Link](https://github.com/AsemFared435/Ai-dermatology-classifier/)
---

## 🌟 Star the Repository

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ by the AI Dermatology Team**
