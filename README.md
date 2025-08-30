# Enhanced Deepfake Detector - Multi-Model Comparison Platform

A cutting-edge web application for detecting deepfakes using multiple Vision Transformer models from Hugging Face, with comprehensive model comparison and performance analysis.

## 🚀 New Features

### Multi-Model Support
- **Custom Model Integration**: Add any Hugging Face ViT model via URL or model name
- **Model Validation**: Automatic validation of model availability and compatibility
- **Dynamic Model Management**: Add/remove models through the web interface
- **Performance Tracking**: Track accuracy, validation, and testing metrics for each model

### Advanced Analysis
- **Single Model Analysis**: Analyze files with your selected model
- **Multi-Model Comparison**: Run the same file through multiple models simultaneously
- **Confidence Scoring**: Detailed confidence levels for each prediction
- **Model Performance Metrics**: Real-time comparison of model effectiveness

### Enhanced Interface
- **Sidebar Navigation**: Streamlined interface with dedicated panels
- **Interactive Charts**: Visual comparison of model performance using Chart.js
- **Real-time Statistics**: Live updates of model database and performance
- **Responsive Design**: Optimized for desktop and mobile devices

## 🎯 Model Performance

### Default Model
- **Model**: `Wvolf/ViT_Deepfake_Detection`
- **Accuracy**: 98.70% on test set
- **Architecture**: Vision Transformer (ViT-Base)
- **Base Model**: google/vit-base-patch16-224

### Recommended Additional Models
- `dima806/deepfake_vs_real_image_detection`
- `rizvandwiki/gender-classification-2`
- `microsoft/DiT-base`
- `nateraw/vit-base-patch16-224-face-detection`

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+
- Python 3.8+
- Internet connection (for model downloads)

### Quick Start
```bash
# 1. Clone and install dependencies
git clone <repository-url>
cd enhanced-deepfake-detector
npm install
pip install -r requirements.txt

# 2. Setup models (choose one)
# Setup recommended models:
python model_setup.py

# Interactive setup:
python model_setup.py --interactive

# Setup specific model:
python model_setup.py --model "username/model-name"

# 3. Start the application
npm run dev
```

Open `http://localhost:3000` in your browser.

## 📁 Project Structure
```
enhanced-deepfake-detector/
├── public/
│   └── index.html              # Enhanced frontend with multi-model UI
├── server.js                   # Enhanced Express server with model management
├── inference.py                # Enhanced Python inference with custom models
├── model_setup.py              # Enhanced setup script for multiple models
├── test_best_model.py          # Testing script
├── package.json                # Updated Node.js dependencies
├── requirements.txt            # Enhanced Python dependencies
├── vercel.json                 # Deployment configuration
└── README.md                   # This documentation
```

## 🔄 API Endpoints

### Model Management
- `GET /api/models` - List all models in database
- `POST /api/models` - Add new model to database
- `DELETE /api/models/:modelName` - Remove model from database

### Analysis
- `POST /api/analyze` - Analyze file with single selected model
- `POST /api/analyze-multiple` - Analyze file with multiple models
- `GET /api/comparison` - Get model comparison data

### System
- `GET /api/health` - Health check with model count
- `GET /api/debug` - System debug information

## 💡 Usage Guide

### Adding Custom Models

1. **Via Web Interface**:
   - Use the sidebar "Add Custom Model" section
   - Enter Hugging Face model URL or name (e.g., `username/model-name`)
   - Click "Add Model"

2. **Via Command Line**:
   ```bash
   python model_setup.py --model "username/model-name"
   ```

3. **Supported Model Formats**:
   - Full URL: `https://huggingface.co/username/model-name`
   - Short form: `username/model-name`

### Model Comparison

1. **Upload Panel**:
   - Select model for single analysis
   - Upload image/video file
   - View results with confidence scores

2. **Comparison Panel**:
   - View performance comparison chart
   - See model statistics and rankings
   - Compare validation vs. testing accuracy

## 🔧 Technical Implementation

### Enhanced Backend Features
- **Model Validation**: Checks model availability on Hugging Face
- **Dynamic Loading**: Loads models on-demand during inference
- **Error Handling**: Graceful fallbacks when models fail to load
- **Performance Tracking**: Stores model metrics in memory database

### Frontend Enhancements
- **Sidebar Navigation**: Clean separation of upload and comparison features
- **Chart.js Integration**: Interactive performance comparison charts
- **Real-time Updates**: Dynamic model list and statistics
- **Responsive Design**: Mobile-friendly interface

### Python Inference Improvements
- **Auto Model Detection**: Supports both Auto and ViT-specific classes
- **Custom Model Support**: Command-line model specification
- **Enhanced Error Handling**: Better error messages and fallbacks
- **Model Information**: Fetches model metadata from Hugging Face API

## 🚀 Deployment

### Vercel (Recommended)
```bash
npm i -g vercel
vercel --prod
```

### Environment Variables
```bash
NODE_ENV=production
```

### Alternative Platforms
- **Railway**: Better for Python ML workloads
- **Google Cloud Run**: Excellent for containerized ML apps
- **Heroku**: Good Python support with buildpacks

## 🎨 Customization

### Adding New Model Categories
Edit the `RECOMMENDED_MODELS` list in `model_setup.py`:
```python
RECOMMENDED_MODELS = [
    'your-model/name',
    'another-user/model-name'
]
```

### Modifying Performance Metrics
Update the placeholder metrics in `server.js`:
```javascript
const performanceMetrics = {
    accuracy: 95.0,  // Your actual accuracy
    validation: 93.0, // Your validation score
    testing: 94.0,   // Your testing score
};
```

### Custom UI Themes
Modify CSS variables in `index.html`:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #48bb78;
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Check internet connection and try:
   python model_setup.py --model "specific-model-name"
   ```

2. **Memory Errors**
   ```bash
   # Use CPU inference or smaller models
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Model Not Found**
   ```bash
   # Verify model exists on Hugging Face:
   curl -I https://huggingface.co/username/model-name
   ```

### Debug Commands
```bash
# Test model validation
python -c "from model_setup import validate_huggingface_model; print(validate_huggingface_model('Wvolf/ViT_Deepfake_Detection'))"

# Test inference
python inference.py path/to/test/image.jpg image "Wvolf/ViT_Deepfake_Detection"

# Check server debug info
curl http://localhost:3000/api/debug
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-model-support`
3. Make changes and test locally
4. Add tests for new functionality
5. Submit a pull request

### Adding New Features
- Model batch processing
- Video frame analysis
- Custom confidence thresholds
- Export comparison reports
- Model performance benchmarking

## 📊 Performance Benchmarks

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for model cache
- **Network**: Stable internet for first-time model downloads

### Expected Performance
- **Single Model Analysis**: 2-5 seconds per image
- **Multi-Model Analysis**: 5-15 seconds (depending on model count)
- **Model Download**: 30-120 seconds (depending on model size)
- **Cold Start**: 1-3 minutes (first request with new model)

## 📄 License

MIT License - Free for educational and commercial use.

## 🙏 Acknowledgments

- **Base Model**: Wvolf/ViT_Deepfake_Detection (Hugging Face)
- **Framework**: Hugging Face Transformers
- **Visualization**: Chart.js
- **Base Architecture**: Google Vision Transformer
- **Community Models**: Various Hugging Face contributors

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open a GitHub issue with detailed information
4. Include debug output from `/api/debug` endpoint

## 🔄 Version History

### v2.0.0 - Multi-Model Support
- ✅ Custom model integration
- ✅ Performance comparison charts
- ✅ Enhanced UI with sidebar navigation
- ✅ Model validation and management
- ✅ Multi-model analysis

### v1.0.0 - Initial Release
- ✅ Single model deepfake detection
- ✅ Basic web interface
- ✅ Image and video support