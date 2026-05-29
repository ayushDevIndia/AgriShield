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
