# FauxFinder

> A multi-model deepfake detection platform powered by Vision Transformers from Hugging Face.

[![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933?logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Express](https://img.shields.io/badge/Express-4.x-000000?logo=express)](https://expressjs.com)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-FFD21E)](https://huggingface.co)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Setup](#model-setup)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Reference](#api-reference)
  - [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

FauxFinder is a web application for detecting AI-generated and manipulated media. It leverages multiple Vision Transformer (ViT) models from Hugging Face, enabling side-by-side model comparison and performance benchmarking on the same input file.

The default model, `Wvolf/ViT_Deepfake_Detection`, achieves **98.70% accuracy** on its test set. Additional models can be added at runtime via the web interface or command line.

---

## Features

| Feature | Description |
|---|---|
| **Multi-Model Analysis** | Run any file through multiple ViT models simultaneously |
| **Custom Model Integration** | Add any Hugging Face image-classification model by URL or name |
| **Model Validation** | Automatic availability and compatibility checks against the HF API |
| **Performance Comparison** | Interactive Chart.js bar charts comparing validation vs. test accuracy |
| **Image & Video Support** | Accepts JPG, PNG, MP4, AVI, and MOV files up to 50 MB |
| **Confidence Scoring** | Per-model softmax probability scores alongside each prediction |
| **Responsive UI** | Sidebar navigation with drag-and-drop upload; works on desktop and mobile |

---

## Demo

```
Upload an image or video → Select one or more models → Get REAL / FAKE predictions with confidence scores
```

A live comparison chart updates in real time as models are added or removed from the model database.

---

## Architecture

```
fauxfinder/
├── public/
│   └── index.html          # Frontend — single-file SPA (Chart.js, vanilla JS)
├── server.js               # Express REST API + Python process orchestration
├── inference.py            # PyTorch inference engine (Auto & ViT class support)
├── model_setup.py          # CLI model downloader and validator
├── test_best_model.py      # Integration test suite
├── requirements.txt        # Python dependencies
├── package.json            # Node.js dependencies
└── vercel.json             # Vercel deployment config
```

**Request flow:**

```
Browser → Express (server.js) → spawn Python (inference.py) → HuggingFace model → JSON result → Browser
```

---

## Getting Started

### Prerequisites

| Dependency | Minimum Version |
|---|---|
| Node.js | 18.0.0 |
| Python | 3.8 |
| pip | latest |
| RAM | 4 GB (8 GB recommended) |
| Disk | 2 GB free (model cache) |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/fauxfinder.git
cd fauxfinder
```

**2. Install Node.js dependencies**

```bash
npm install
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

### Model Setup

Choose one of the following setup options:

```bash
# Download all recommended models
python model_setup.py

# Interactive mode — choose which models to install
python model_setup.py --interactive

# Download a specific model
python model_setup.py --model "username/model-name"
```

**Recommended models:**

| Model | Notes |
|---|---|
| `Wvolf/ViT_Deepfake_Detection` | Default — 98.70% test accuracy |
| `dima806/deepfake_vs_real_image_detection` | Alternative deepfake detector |
| `microsoft/DiT-base` | Document Image Transformer baseline |

**4. Start the development server**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Usage

### Web Interface

#### Uploading a file

1. Navigate to the **Upload & Analyze** tab.
2. Select your model from the dropdown (or add a custom one first).
3. Drag and drop, or click **Choose File**, to upload an image or video.
4. Results appear below the preview with a prediction label and confidence score.

#### Adding a custom model

1. Paste a Hugging Face model URL or short-form name into the **Add Custom Model** field in the sidebar.
   - Full URL: `https://huggingface.co/username/model-name`
   - Short form: `username/model-name`
2. Click **Add Model**. The server validates the model against the Hugging Face API before adding it.

#### Comparing models

Switch to the **Model Comparison** tab to see a bar chart of validation vs. test accuracy across all registered models, along with aggregate statistics.

---

### API Reference

#### Model Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all models in the database |
| `POST` | `/api/models` | Add a new model |
| `DELETE` | `/api/models/:modelName` | Remove a model |

**POST `/api/models` — request body:**

```json
{
  "modelName": "username/model-name"
}
```

#### Analysis

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Analyse a file with one model |
| `POST` | `/api/analyze-multiple` | Analyse a file with multiple models |
| `GET` | `/api/comparison` | Retrieve model comparison data |

**POST `/api/analyze` — form data:**

| Field | Type | Description |
|---|---|---|
| `file` | `File` | Image or video file (max 50 MB) |
| `model` | `string` | Hugging Face model name |

**Response:**

```json
{
  "prediction": "FAKE",
  "confidence": 0.9741,
  "file_type": "image",
  "status": "success",
  "model": "Wvolf/ViT_Deepfake_Detection",
  "model_info": {}
}
```

#### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/debug` | Runtime environment info |

---

### CLI Usage

Run inference directly from the command line:

```bash
# Analyse an image with the default model
python inference.py path/to/image.jpg image

# Analyse a video with a custom model
python inference.py path/to/video.mp4 video "username/model-name"
```

Validate a model without downloading it:

```bash
python -c "from model_setup import validate_huggingface_model; print(validate_huggingface_model('Wvolf/ViT_Deepfake_Detection'))"
```

Run the integration test suite:

```bash
npm run test-models
# or
python test_best_model.py
```

---

## Configuration

### Adding recommended models

Edit `RECOMMENDED_MODELS` in `model_setup.py`:

```python
RECOMMENDED_MODELS = [
    'Wvolf/ViT_Deepfake_Detection',
    'your-username/your-model',
]
```

### Adjusting placeholder performance metrics

Update the defaults in `server.js` for seeded model data:

```javascript
const performanceMetrics = {
    accuracy: 95.0,
    validation: 93.0,
    testing: 94.0,
};
```

### Theming

CSS custom properties are defined at the top of `public/index.html`:

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #48bb78;
}
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | HTTP server port |
| `NODE_ENV` | `development` | Runtime environment |

---

## Deployment

### Vercel (recommended for front-end + API)

```bash
npm i -g vercel
vercel --prod
```

The included `vercel.json` routes all `/api/*` requests to `server.js` and serves `public/` as static assets.

> **Note:** Vercel's serverless functions have a maximum execution duration of 30 seconds. Cold-start model loading may exceed this for large models. For production ML workloads, consider the platforms below.

### Alternative platforms

| Platform | Notes |
|---|---|
| **Railway** | Native Python support; good for persistent model caching |
| **Google Cloud Run** | Containerised deployment; scale-to-zero |
| **Heroku** | Python buildpacks available; requires dyno with sufficient RAM |

---

## Troubleshooting

### Model download fails

```bash
# Verify the model exists on Hugging Face
curl -I https://huggingface.co/username/model-name

# Retry with explicit model name
python model_setup.py --model "username/model-name"
```

### Out-of-memory errors

```bash
# Force CPU-only inference
export CUDA_VISIBLE_DEVICES=""
python inference.py path/to/file.jpg image
```

### No JSON output from inference script

```bash
# Check server debug info
curl http://localhost:3000/api/debug
```

### Python not found on Windows

Ensure Python is on your `PATH`, or update the `pythonCmd` variable in `server.js`:

```javascript
const pythonCmd = 'python'; // Windows
```

---

## Contributing

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and add tests where applicable.
3. Ensure existing tests pass:
   ```bash
   python test_best_model.py
   ```
4. Open a pull request with a clear description of your changes.

**Ideas for contributions:**

- Batch file processing
- Per-frame video analysis with timeline view
- Exportable comparison reports (CSV / PDF)
- Custom confidence threshold configuration
- Docker / docker-compose setup

---

## Acknowledgements

- [Wvolf/ViT_Deepfake_Detection](https://huggingface.co/Wvolf/ViT_Deepfake_Detection) — primary detection model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — model loading and inference
- [Google Vision Transformer](https://arxiv.org/abs/2010.11929) — underlying ViT architecture
- [Chart.js](https://www.chartjs.org) — performance comparison visualisations
