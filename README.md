<h1 align="center">
  <br>
  <img src="assets/banner.png" alt="DigitVision AI" width="900">
  <br>
  DigitVision AI — Handwritten Digit Recognition
  <br>
</h1>

<h4 align="center">A production-ready deep learning application that recognizes handwritten digits using a custom Convolutional Neural Network trained on the MNIST dataset — featuring a stunning real-time interactive web interface.</h4>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-2.21%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-1.32%2B-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
  <a href="https://opencv.org/">
    <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License">
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-model-performance">Performance</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎨 **Draw Mode** | Real-time canvas — draw a digit and get instant predictions |
| 📤 **Upload Mode** | Upload any digit image (PNG, JPG, JPEG) for recognition |
| 🧠 **CNN Model** | Custom 4-conv-block architecture trained on 60,000 MNIST images |
| 📊 **Probability Bars** | Animated confidence visualization for all 10 digit classes |
| 🔄 **Smart Preprocessing** | Auto-background detection, contour cropping, MNIST-style normalization |
| 🎨 **Premium UI** | Dark glassmorphic design with smooth animations and gradient accents |
| 🖥️ **Pipeline Preview** | Step-by-step visualization of the image preprocessing pipeline |
| ⚡ **Fast Inference** | Sub-100ms predictions with TensorFlow-CPU |

---

## 🏗️ Architecture

### CNN Model

```
Input (28×28×1)
    │
    ├── Conv2D(32, 3×3, relu) + BatchNorm + Conv2D(32, 3×3, relu)
    ├── MaxPooling2D(2×2) + Dropout(0.25)
    │
    ├── Conv2D(64, 3×3, relu) + BatchNorm + Conv2D(64, 3×3, relu)
    ├── MaxPooling2D(2×2) + Dropout(0.25)
    │
    ├── Flatten → Dense(128, relu) + Dropout(0.3)
    │
    └── Dense(10, softmax) → Predicted Digit (0–9)
```

**Training Config:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Batch Size: 128 | Max Epochs: 5

### Preprocessing Pipeline

```
Raw Input
    → Grayscale Conversion
    → Border Analysis (auto-detect white/dark background)
    → Bitwise Inversion (if needed, to match MNIST style)
    → Otsu Thresholding
    → Contour Detection & Crop
    → Aspect-Ratio-Preserving Pad + Center
    → Resize to 28×28
    → Normalize [0, 1]
    → Reshape to (1, 28, 28, 1)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) GPU with CUDA for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/muhammadabdullah-devpk/CodeAlpha_Handwritten_Character_Recognition.git
cd CodeAlpha_Handwritten_Character_Recognition
```

### 2. Create & Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

> Training downloads the MNIST dataset automatically on first run.  
> The trained model is saved to `models/best_model.keras`.

### 5. Launch the Web App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🎯 CLI Usage

**Predict from a custom image:**

```bash
python predict.py --image assets/sample_digit.png
```

**Evaluate the saved model on the MNIST test set:**

```bash
python evaluate.py
```

---

## 📂 Project Structure

```
CodeAlpha_Handwritten_Character_Recognition/
│
├── app.py                  # 🌐 Streamlit web application (main entry point)
├── train.py                # 🏋️ Model training script
├── predict.py              # 🔍 CLI prediction script
├── evaluate.py             # 📋 Model evaluation script
├── requirements.txt        # 📦 Python dependencies
│
├── src/                    # 🧩 Core library modules
│   ├── __init__.py
│   ├── config.py           # Path constants and project configuration
│   ├── data_loader.py      # MNIST data loading and splitting
│   ├── model_builder.py    # CNN architecture, training, and loading
│   ├── preprocessing.py    # OpenCV image preprocessing pipeline
│   └── utils.py            # Plotting, JSON/text reporting utilities
│
├── models/                 # 💾 Saved model artifacts
│   └── best_model.keras    # Trained CNN model (auto-generated by train.py)
│
├── assets/                 # 🖼️ Static assets and sample images
│   └── sample_digit.png    # Sample image for CLI testing
│
└── reports/                # 📊 Auto-generated training reports
    ├── training_history.png
    ├── training_summary.json
    └── training_summary.txt
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Dataset | MNIST (60K train / 10K test) |
| Test Accuracy | **~99%** |
| Test Loss | **~0.03** |
| Inference Time | < 50ms per image |
| Model Size | ~2.5 MB |

> Exact metrics are saved to `reports/training_summary.json` after training.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras |
| **Image Processing** | OpenCV, Pillow |
| **Web Framework** | Streamlit |
| **Data Science** | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| **Canvas Drawing** | streamlit-drawable-canvas |

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun et al.
- [TensorFlow/Keras](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the interactive web framework
- [CodeAlpha](https://www.codealpha.tech/) — internship project context

---

<p align="center">
  Made with ❤️ by <strong>Muhammad Abdullah</strong>
</p>
