# 🌾 Rice Leaf Disease Detection using Deep Learning

An end-to-end Deep Learning project for automatic **Rice Leaf Disease Classification** using **EfficientNet-B0** with an interactive **Streamlit Web Application**. The system predicts rice leaf diseases from uploaded images and provides detailed disease information, symptoms, causes, treatments, and prevention measures.

---

## 📌 Features

- 🔍 Automatic Rice Leaf Disease Detection
- 🧠 EfficientNet-B0 Transfer Learning
- 📊 Model Evaluation (Accuracy, Precision, Recall, F1-Score)
- 📈 Confusion Matrix & Classification Report
- 🔥 Grad-CAM Explainability
- 💡 Disease Advisory System
- 🌐 Interactive Streamlit Web App
- 📂 Modular Project Structure
- 💾 Best Model Checkpoint Saving & Loading

---

## 🦠 Diseases Supported

- Bacterial Blight
- Blast
- Brown Spot
- Tungro

---

## 🏗️ Project Architecture

```
Rice Leaf Image
        │
        ▼
Image Preprocessing
        │
        ▼
EfficientNet-B0
        │
        ▼
Disease Prediction
        │
        ▼
Confidence Score
        │
        ▼
Disease Knowledge Base
        │
        ▼
Treatment & Prevention
        │
        ▼
Grad-CAM Visualization
```

---

# 📂 Project Structure

```
CropDisease/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_model.pth
│
├── knowledge_base/
│   └── rice_disease_info.json
│
├── notebooks/
│   ├── Data_Inspection.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   ├── Evaluation.ipynb
│   ├── GradCam.ipynb
│   ├── Advisory_System.ipynb
│   └── Final_Report.ipynb
│
├── reports/
│
├── outputs/
│
└── src/
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    ├── transforms.py
    └── utils.py
```

---

# 🚀 Tech Stack

### Deep Learning

- PyTorch
- Torchvision
- TIMM
- EfficientNet-B0

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Explainability

- Grad-CAM

### Web Framework

- Streamlit

---

# 🧠 Model

**Architecture**

EfficientNet-B0 (Transfer Learning)

- Image Size: 224 × 224
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Early Stopping
- Best Model Checkpoint Saving

---

# 📊 Performance

| Metric | Score |
|---------|-------|
| Test Accuracy | **99.33%** |
| Precision | High |
| Recall | High |
| F1 Score | High |

> Performance may vary depending on dataset split.

---

# 🔥 Explainable AI

The project integrates **Grad-CAM** to visualize which regions of the rice leaf influenced the model's prediction.

This improves:

- Transparency
- Trustworthiness
- Model Interpretability

---

# 💡 Disease Advisory System

For every prediction, the application provides:

- Disease Description
- Symptoms
- Causes
- Treatment
- Prevention
- Recommended Agricultural Practices

---

# 🌐 Streamlit Application

The web application allows users to:

- Upload a rice leaf image
- Predict disease instantly
- View prediction confidence
- Read disease information
- Visualize Grad-CAM heatmaps *(optional extension)*

Run locally:

```bash
streamlit run app.py
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Rice-Leaf-Disease-Detection.git
```

Move into the project directory

```bash
cd Rice-Leaf-Disease-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Demo

### Upload Image

✔ Rice Leaf Image

↓

### Model Prediction

✔ Disease Name

✔ Confidence Score

↓

### Disease Advisory

✔ Description

✔ Symptoms

✔ Causes

✔ Treatment

✔ Prevention

---

# Future Improvements

- 🌦 Weather-based Disease Prediction
- 📍 GPS Location Support
- ☁ Cloud Deployment
- 📱 Mobile Application
- 📷 Real-Time Camera Detection
- 🤖 Multi-Disease Detection
- 🌿 Pest Identification
- 📊 Disease Severity Estimation
- 🔥 Integrated Grad-CAM in Streamlit

---

# 👨‍💻 Author

**Shlok Nandwana**

B.Tech Information Technology

Artificial Intelligence | Machine Learning | Deep Learning | Computer Vision

GitHub: https://github.com/shlok0603

LinkedIn: https://www.linkedin.com/in/shlok-nandwana

---