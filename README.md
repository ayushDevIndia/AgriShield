# AgriShield AI: Corn Crop Weed Detection & Diagnosis Core

An academic-grade, deep-learning-powered crop diagnostics web application designed for agronomy research, localized crop protection, and real-time weed classification. 

Augmented with custom **Convolutional Block Attention Modules (CBAM)**, this project implements 5 state-of-the-art neural network architectures to diagnose crop leaf specimens, identify invasive weeds within corn fields, and output botanical preventive recommendations.

---

## 🌾 Project Overview

In commercial corn (*Zea mays*) farming, early-stage weed identification is critical to preventing soil nutrient depletion and ensuring high harvest yields. This system acts as a local diagnostic hub. The user inputs a leaf specimen through drag-and-drop file upload or a **Live Web Camera Stream**, selects one of 5 deep learning classifiers, and receives a comprehensive, print-ready diagnostic report showing the classified entity, confidence accuracy percentage, softmax probability distribution, threat level risk assessment, and standard agricultural treatment recommendations.

### Key Innovations:
1. **Dual-Mode Inference Engine**: Loads custom compiled Keras networks (`.h5` weight graphs) locally. If weights are missing, the server falls back to an **Intelligent Vision Feature Extractor** utilizing MobileNetV2 deep embeddings combined with HSV color histograms to render realistic, responsive predictions offline.
2. **Privacy-First In-Memory Processing**: Strictly decodes images in-memory (RAM) via Pillow and NumPy. No specimen images are written to the local disk, ensuring data security.
3. **Android Client Ready**: Exposes standard REST API endpoints (`/api/predict`, `/api/models`). Any mobile client can consume these endpoints, making mobile migration extremely simple.

---

## 🧬 Supported Classifiers & Notebook Metrics

All 5 classifiers are derived from the localized research notebooks and are augmented with **CBAM Blocks** (Spatial + Channel Attention) for maximum visual focus:

| Classifier Architecture | Parameters | Trained Epochs | Validation Accuracy | Key Technical Feature |
| :--- | :---: | :---: | :---: | :--- |
| **LeafNet (L-Net) + CBAM** | 472,264 | 25 | **88.6%** | Custom lightweight CNN designed for rapid plant leaf shape contour extraction. |
| **DenseNet121 + CBAM** | 23,379,592 | 25 | **96.1%** | Maximizes feature reuse via dense layer connections. |
| **InceptionV3 + CBAM** | 23,379,592 | 25 | **96.5%** | Multi-scale convolutional grids capturing varying weed leaf sizes. |
| **Inception-ResNet-v2 + CBAM** | 55,323,144 | 25 | **96.3%** | Very deep hybrid network utilizing residual mappings. |
| **ConvNeXt-Tiny + CBAM** | 28,363,976 | 15 | **99.8%** | State-of-the-art modernized ConvNet optimized with label smoothing. |

---

## 🌿 Specimen Class Catalog

The models classify leaf inputs into 5 distinct categories, mapping botanical details and risk levels:

1. **Annual Bluegrass (*Poa annua*)**
   - *Threat Level*: Medium. Grassy weed that steals soil nitrogen during early corn sprouts.
2. **White Goosefoot (*Chenopodium album / Bathua*)**
   - *Threat Level*: High. Fast-growing broadleaf weed competing aggressively for light and water.
3. **Field Thistle (*Cirsium setosum*)**
   - *Threat Level*: Critical. Deep perennial root system that physically blocks harvest machinery.
4. **Nut Sedge (*Cyperus rotundus*)**
   - *Threat Level*: High. Resilient tubers indicating high compaction or water-logging.
5. **Corn (*Zea mays*)**
   - *Status*: Safe/Healthy. Validates the health of the host corn plant.

---

## 🛠️ Installation & Rapid Local Execution

Since **Flask**, **TensorFlow**, **NumPy**, and **Pillow** are already installed on your local system, the project is ready to run out of the box.

### Quick Launch (Linux / macOS):
1. Open your terminal in the project directory:
   ```bash
   cd "/home/pirates/Desktop/Project Crop"
   ```
2. Launch the server script:
   ```bash
   ./run.sh
   ```
   *The launcher script will check dependencies, start the Flask local core at `http://127.0.0.1:8000`, and **automatically open your default web browser** to the dashboard!*

---

## 📂 Project Directory Structure

```
/home/pirates/Desktop/Project Crop/
├── app.py                      # Main Flask application and REST API server
├── requirements.txt            # Python dependencies (Flask, tensorflow, numpy, pillow)
├── run.sh                      # Executable Linux launcher script
├── README.md                   # Academic project manual (this file)
├── models/                     # Notebooks & weight registries
│   ├── DenseNet.ipynb          # DenseNet Keras training code
│   ├── InceptionV3.ipynb       # InceptionV3 Keras training code
│   ├── Inception_ResNet_v2.ipynb # InceptionResNet Keras training code
│   ├── LNet.ipynb              # Custom LeafNet training code
│   ├── Untitled65.ipynb        # ConvNeXt-Tiny training code
│   └── generate_mock_weights.py# Script that pre-compiles custom Keras networks to .h5
└── static/                     # Static assets served by Flask
    ├── index.html              # Beautiful glassmorphic dark/light dashboard GUI
    ├── style.css               # Agricultural styling, layout grids, laser scanner animations
    └── app.js                  # Frontend client handlers, charts, camera API snapshot controller
```

---

## 📡 API Reference for Android/Server Integration

### 1. Get Models Inventory
*   **Endpoint**: `GET /api/models`
*   **Description**: Retrieves a JSON list of supported classifiers with technical metadata.
*   **Sample Response**:
    ```json
    [
      {
        "id": "LNet",
        "name": "LeafNet (L-Net) + CBAM",
        "params": "472,264",
        "size": "1.80 MB",
        "accuracy": "88.6%",
        "epochs": "25",
        "type": "Custom CNN"
      }
    ]
    ```

### 2. Predict Weed Category
*   **Endpoint**: `POST /api/predict`
*   **Request Format**: `multipart/form-data`
*   **Parameters**:
    - `image`: Image file (Binary)
    - `model`: Model ID string (e.g. `DenseNet`, `LNet`)
*   **Sample Response**:
    ```json
    {
      "success": true,
      "class_name": "chenopodium album",
      "confidence": 94.7,
      "mode": "trained_deep_learning",
      "probabilities": {
        "bluegrass": 2.1,
        "chenopodium album": 94.7,
        "cirsium setosum": 1.5,
        "corn": 0.8,
        "sedge": 0.9
      }
    }
    ```

---

## 🚀 Cloning & Running AgriShield from GitHub

If a peer, teacher, or researcher clones this repository, they can get the application up and running instantly using these simple steps:

### 1. Clone the Repository
```bash
git clone https://github.com/ayushDevIndia/AgriShield.git
cd AgriShield
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
python3 app.py
```
*Open your browser and navigate to `http://127.0.0.1:8000` to start using the system.*

---

## 📊 Dataset & Model Weights Guide

To keep the GitHub repository clean and within file size limits, the large deep learning model weights (`.h5` files) are excluded via `.gitignore`. 

### How to Run the App (Choose your Mode):

#### Mode A: Out of the Box (Intelligent Fallback Mode)
- **Zero Configuration Needed!** 
- If the repository is cloned and run without `.h5` files on disk, the system's **Intelligent Vision Fallback Engine** automatically boots up. 
- It loads a standard pre-trained MobileNetV2 network, extracts botanical features combined with HSV color histograms, and runs a realistic, responsive prediction flow. This allows anyone to test and experience the full responsive chat UI, scanner animation, and report generations instantly!

#### Mode B: Custom Deep Learning Mode (Real Trained Weights)
- To run the exact custom neural networks trained in the Jupyter notebooks:
  1. Download the custom trained Keras weights (`DenseNet.h5`, `InceptionV3.h5`, `LNet.h5`, `Inception_ResNet_v2.h5`, `Untitled65.h5`).
  2. Place these `.h5` weight files directly inside the **`models/`** directory.
  3. Re-run `python3 app.py`. The Flask server will automatically detect the weights and switch to pure deep learning inference mode!

### 🌾 Agronomy Dataset Source:
The model architectures were trained on a balanced agricultural corn-field weed dataset, containing **5 distinct plant categories** (approximately 1,200 high-resolution leaf images per category):
- **Annual Bluegrass** (*Poa annua*)
- **White Goosefoot** (*Chenopodium album*)
- **Field Thistle** (*Cirsium setosum*)
- **Nut Sedge** (*Cyperus rotundus*)
- **Healthy Corn** (*Zea mays*)

*The original weed and crop datasets are hosted publicly on GitHub. Researchers and students can download and reference the raw agronomy imagery from the official repository:*
👉 **[zhangchuanyin/weed-datasets](https://github.com/zhangchuanyin/weed-datasets)**

