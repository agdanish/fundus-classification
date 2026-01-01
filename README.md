# 👁️ Fundus Image Classification App

A Streamlit-based web application for automatic classification of fundus (retinal) images as **Normal** or **Abnormal** using deep learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

## 🌟 Features

- **Easy Upload**: Simply upload fundus images in common formats (JPG, PNG, etc.)
- **Real-time Classification**: Get instant predictions on image normality
- **Confidence Visualization**: View prediction confidence scores with interactive bar charts
- **Deep Learning Powered**: Built on InceptionV3 architecture with transfer learning
- **User-Friendly Interface**: Clean and intuitive Streamlit interface

## 🎯 Live Demo

🔗 **[Try it here]** (Add your deployment link)

## 🏗️ Architecture

- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Input Size**: 150x150 pixels
- **Classes**: 2 (Normal, Abnormal)
- **Framework**: TensorFlow/Keras
- **Frontend**: Streamlit

## 📊 Model Information

- Pre-trained InceptionV3 layers: 311
- Custom classification layers
- Trained on fundus image dataset
- Optimized for binary classification

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/fundus-classification.git
cd fundus-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run fundus.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📁 Project Structure

```
fundus-classification/
│
├── fundus.py              # Main Streamlit application
├── fun.h5                 # Trained model weights
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
└── img/                  # Image assets
    ├── 1.jfif           # App icon
    └── 5.jpg            # Header image
```

## 🎮 How to Use

1. Launch the application
2. Click on "UPLOAD IMAGE" in the sidebar
3. Select a fundus image from your computer
4. Click the "🔄 PREDICT" button
5. View the classification result and confidence scores

## 📸 Sample Input

The app accepts fundus (retinal) images in the following formats:
- JPG/JPEG
- PNG
- Other common image formats

## 📈 Results

The application provides:
- **Primary Prediction**: Normal or Abnormal classification
- **Confidence Score**: Percentage confidence for each class
- **Visualization**: Bar chart showing prediction probabilities
- **Processing Time**: Time taken for prediction

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Deep Learning | TensorFlow 2.18 |
| Base Model | InceptionV3 |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |

## ⚙️ Configuration

Key parameters in `fundus.py`:
- `target_size = (150, 150)` - Input image size
- `class_names = ['AbNormal', 'Normal']` - Classification labels

## 🔧 Development

### Running Locally

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run fundus.py
```

### Model Training

The model (`fun.h5`) was trained using:
- Transfer learning with InceptionV3
- Custom top layers for binary classification
- Adam optimizer
- Categorical crossentropy loss

## 📝 Notes

- The model file (`fun.h5`) is loaded with `compile=False` for compatibility with different TensorFlow versions
- Model is recompiled at runtime with modern Keras settings
- Some TensorFlow warnings during startup are normal and don't affect functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - Danish A G

## 🙏 Acknowledgments

- InceptionV3 architecture by Google
- Streamlit framework for the amazing UI tools
- TensorFlow team for the deep learning framework

## 📞 Contact

For questions or feedback:
- Create an issue in this repository
- Email: danish@xzashr.com

## 🔮 Future Enhancements

- [ ] Multi-class classification (different disease types)
- [ ] Grad-CAM visualization for interpretability
- [ ] User authentication and history tracking
- [ ] Batch processing for multiple images
- [ ] PDF report generation
- [ ] Integration with medical databases
- [ ] Mobile app version

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

Made with ❤️ using Streamlit and TensorFlow
=======
# fundus-classification
Deep learning fundus image classifier built with TensorFlow InceptionV3 and Streamlit. Classifies retinal images as Normal or Abnormal with confidence scores.
