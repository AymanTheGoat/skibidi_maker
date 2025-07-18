# 🚽 Skibidi Face Generator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AymanTheGoat/skibidi_maker/blob/main/LICENSE)

> **Transform faces into epic Skibidi toilet memes!**

## 🎭 Overview

The Skibidi Face Generator is a Python-based application that uses computer vision to attach head of a person in input image into toilet to create Skibidi toilet meme

### ✨ Key Features

- **🔍 Dual Face Detection**: Choose between HaarCascades and ULFGFD (Ultra Light Fast Generic Face Detector) models
- **🚽 Automatic Composition**: Everything is automatic (except the initial setup! (for now))
- **📸 Batch Processing**: Coming soon

## 🖼️ Preview

<!-- Add your preview images here -->
![Demo](https://github.com/AymanTheGoat/skibidi_maker/raw/main/assets/demo.png)


## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **OpenCV**: Computer vision and image processing
- **ONNX Runtime**: High-performance inference engine

### ML Models
- **U2NET**: Background removal and segmentation
- **ULFGFD**: Ultra-light generic face detection (First option)
- **Haar Cascade**: Traditional face detection (Second option)

## 📁 Project Structure

```
skibidi-face-generator/
├── 📄 main.py                    # Main application entry point
├── 📁 utils/                     # Utility modules
│   ├── 🔍 ULFGFD_detection.py   # Advanced face detection
│   ├── 🔍 haar_detection.py     # Traditional face detection
│   ├── 🎨 image_utils.py        # Image processing utilities
│   ├── 🗑️ remove_bg.py          # Background removal
│   ├── 📁 file_utils.py         # File operations
│   └── 📊 logger.py             # Logging system
├── 📁 assets/                    # Static assets
│   ├── 🚽 toilet.png            # Base toilet image
│   └── 🎭 toilet_overlay.png    # Overlay elements
├── 📁 weights/                   # AI model files
│   ├── 🧠 u2net.onnx            # Background removal model
│   ├── 👤 version-RFB-640.onnx  # Face detection model
│   └── 📊 haarcascade.xml       # Haar cascade classifier
├── 📁 input/                     # Input images
├── 📁 output/                    # Generated results
└── 📖 README.md                 # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher

### Dependencies
```bash
pip install opencv-python onnxruntime numpy
```

### Model Files
Download the required model files and place them in the `weights/` directory:
- `u2net.onnx` - U2NET background removal model
- `version-RFB-640.onnx` - ULFGFD face detection model
- `haarcascade.xml` - Haar cascade classifier

## 🎯 Features Deep Dive

### 🔍 Machine Learning Models
- **ULFGFD (Ultra Light Fast Generic Face Detector)**: For face detection, slower and more accurate 
- **HaarCascades**: For Face detection, faster and less accurate 
- **U2net**: For backhround removal, very fast and accurate 

### 🚽 Composition System
- **Positioning**: Automatically resizes and centers faces in toilet bowls in a preset place
- **Optimization**: Super fast throughout the entire process

## 🔧 Configuration

### Detection Method Selection
```python
method = 1  # Haar Cascade
method = 2  # ULFGFD (recommended)
```

### Confidence Thresholds
```python
confidence_threshold = 0.5  # Adjust for detection sensitivity
```

### Output Settings
```python
# Do not mess with these unless you know what you are doing
SIZE = 1080      # Canvas size
WIDTH = 383      # Face width
HEIGHT = 513     # Face height
X = 105          # X position
Y = 215          # Y position
```

## 🐛 Troubleshooting

### Common Issues

**No faces detected**
- Ensure good lighting in input images
- Try adjusting confidence threshold
- Switch between detection methods

**Weights not found**
- Ensure all weights are downloaded, and weights paths are correct 

## 📊 Logging System

The application features a comprehensive logging system with:
- **Color-coded 3-level loging**: Info (Blue), Warning (Yellow), Error (Red)
- **On error**: exit

## 🔮 Future Enhancements

- [ ] Web interface for easy usage
- [ ] Additional meme templates
- [ ] Batch processing interface
- [ ] GPU acceleration support
- [ ] Real-time video processing

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more details.

### Development Setup
1. Make an issue or contact me (contacts below)
2. Fork the repository
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[U2NET](https://github.com/xuebinqin/U-2-Net)**: For efficient background removal model
- **[Ultra Light Fast Generic Face Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)**: For lightweight face detection
- **[OpenCV](https://github.com/opencv/opencv)**: For comprehensive computer vision library
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)**: For high-performance inference engine

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/AymanTheGoat/skibidi_maker/issues)
- 📧 **Contact**: Discord: ".wp5" , X: [@aymandotexe](https://x.com/aymandotexe), Instagram: [@ayman_the binladen](https://www.instagram.com/ayman_the_binladen/)

---

<div align="center">
  <p><strong>Made with ❤️ and lots of 🤣🤣</strong></p>
  <p><em>Bringing meme culture and computer vision together, one Skibidi face at a time!</em></p>
</div>
