// Enhanced server.js with multi-model support and comparison functionality
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// In-memory storage for model performance data
let modelDatabase = {
    'Wvolf/ViT_Deepfake_Detection': {
        name: 'Wvolf/ViT_Deepfake_Detection',
        accuracy: 98.7,
        validation: 97.2,
        testing: 98.7,
        downloads: 3088,
        status: 'active',
        addedAt: new Date().toISOString()
    }
};

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = './uploads';
        fs.ensureDirSync(uploadDir);
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        const ext = path.extname(file.originalname);
        cb(null, `upload_${timestamp}${ext}`);
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|mp4|avi|mov/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only images and videos are allowed.'));
        }
    }
});

// Utility function to validate Hugging Face model
async function validateHuggingFaceModel(modelName) {
    try {
        // Clean model name
        let cleanName = modelName.includes('huggingface.co/') 
            ? modelName.split('huggingface.co/')[1] 
            : modelName;
        
        if (cleanName.startsWith('/')) {
            cleanName = cleanName.substring(1);
        }

        // Check if model exists
        const response = await axios.head(`https://huggingface.co/${cleanName}`, {
            timeout: 10000
        });
        
        if (response.status === 200) {
            // Try to get model info
            try {
                const infoResponse = await axios.get(`https://huggingface.co/api/models/${cleanName}`, {
                    timeout: 10000
                });
                
                return {
                    valid: true,
                    name: cleanName,
                    info: infoResponse.data
                };
            } catch (infoError) {
                // Model exists but no API info available
                return {
                    valid: true,
                    name: cleanName,
                    info: null
                };
            }
        }
        
        return {
            valid: false,
            error: `Model not found on Hugging Face`
        };
        
    } catch (error) {
        return {
            valid: false,
            error: `Validation failed: ${error.message}`
        };
    }
}

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Get all models in database
app.get('/api/models', (req, res) => {
    const models = Object.values(modelDatabase);
    res.json({
        models,
        total: models.length,
        status: 'success'
    });
});

// Add new model to database
app.post('/api/models', async (req, res) => {
    try {
        const { modelName } = req.body;
        
        if (!modelName) {
            return res.status(400).json({ error: 'Model name is required' });
        }

        console.log(`Adding new model: ${modelName}`);
        
        // Validate model
        const validation = await validateHuggingFaceModel(modelName);
        
        if (!validation.valid) {
            return res.status(400).json({ 
                error: validation.error,
                status: 'validation_failed'
            });
        }

        const cleanName = validation.name;

        // Check if already exists
        if (modelDatabase[cleanName]) {
            return res.status(409).json({ 
                error: 'Model already exists in database',
                status: 'already_exists'
            });
        }

        // Generate placeholder performance metrics (in real app, these would come from actual testing)
        const performanceMetrics = {
            accuracy: Math.random() * 10 + 88, // 88-98%
            validation: Math.random() * 10 + 85, // 85-95%
            testing: Math.random() * 10 + 87, // 87-97%
        };

        // Add to database
        modelDatabase[cleanName] = {
            name: cleanName,
            ...performanceMetrics,
            downloads: validation.info?.downloads || 0,
            likes: validation.info?.likes || 0,
            status: 'active',
            addedAt: new Date().toISOString(),
            library: validation.info?.library_name || 'transformers',
            pipeline_tag: validation.info?.pipeline_tag || 'image-classification'
        };

        console.log(`Model ${cleanName} added successfully`);
        
        res.json({
            message: 'Model added successfully',
            model: modelDatabase[cleanName],
            status: 'success'
        });

    } catch (error) {
        console.error('Error adding model:', error);
        res.status(500).json({ 
            error: 'Failed to add model',
            details: error.message
        });
    }
});

// Remove model from database
app.delete('/api/models/:modelName', (req, res) => {
    try {
        const modelName = decodeURIComponent(req.params.modelName);
        
        if (!modelDatabase[modelName]) {
            return res.status(404).json({ error: 'Model not found' });
        }

        // Don't allow deletion of default model
        if (modelName === 'Wvolf/ViT_Deepfake_Detection') {
            return res.status(400).json({ error: 'Cannot delete default model' });
        }

        delete modelDatabase[modelName];
        
        res.json({
            message: 'Model removed successfully',
            status: 'success'
        });

    } catch (error) {
        console.error('Error removing model:', error);
        res.status(500).json({ error: 'Failed to remove model' });
    }
});

// Analyze file with single model
app.post('/api/analyze', upload.single('file'), async (req, res) => {
    try {
        console.log('=== SINGLE MODEL ANALYSIS START ===');
        
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const filePath = path.resolve(req.file.path);
        const isVideo = req.file.mimetype.startsWith('video/');
        const selectedModel = req.body.model || 'Wvolf/ViT_Deepfake_Detection';

        console.log(`Analyzing with model: ${selectedModel}`);
        console.log(`File: ${req.file.originalname} (${isVideo ? 'video' : 'image'})`);

        // Check if inference.py exists
        const inferenceScript = path.join(__dirname, 'inference.py');
        if (!fs.existsSync(inferenceScript)) {
            throw new Error('Inference script not found');
        }

        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        
        const result = await new Promise((resolve, reject) => {
            const pythonProcess = spawn(pythonCmd, [
                'inference.py', 
                filePath, 
                isVideo ? 'video' : 'image',
                selectedModel
            ], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe']
            });
            
            let stdout = '';
            let stderr = '';
            
            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout: stdout.trim(), stderr });
                } else {
                    reject(new Error(`Analysis failed with code ${code}\nStderr: ${stderr}`));
                }
            });
            
            pythonProcess.on('error', (err) => {
                reject(err);
            });
        });

        // Parse JSON result
        const lines = result.stdout.split('\n').filter(line => line.trim().length > 0);
        let jsonLine = lines.find(line => line.trim().startsWith('{'));
        
        if (!jsonLine) {
            throw new Error('No JSON output found');
        }
        
        const parsedResult = JSON.parse(jsonLine.trim());
        
        // Clean up
        fs.removeSync(filePath);
        
        console.log('=== SINGLE MODEL ANALYSIS END ===');
        res.json(parsedResult);

    } catch (error) {
        console.error('Analysis error:', error);
        
        // Clean up file
        if (req.file && req.file.path) {
            try {
                fs.removeSync(req.file.path);
            } catch (cleanupError) {
                console.error('Failed to cleanup file:', cleanupError);
            }
        }
        
        res.status(500).json({ 
            error: 'Analysis failed',
            details: error.message
        });
    }
});

// Analyze file with multiple models for comparison
app.post('/api/analyze-multiple', upload.single('file'), async (req, res) => {
    try {
        console.log('=== MULTI-MODEL ANALYSIS START ===');
        
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const filePath = path.resolve(req.file.path);
        const isVideo = req.file.mimetype.startsWith('video/');
        const modelsToTest = req.body.models ? JSON.parse(req.body.models) : Object.keys(modelDatabase);

        console.log(`Analyzing with ${modelsToTest.length} models`);
        console.log(`Models: ${modelsToTest.join(', ')}`);

        const inferenceScript = path.join(__dirname, 'inference.py');
        if (!fs.existsSync(inferenceScript)) {
            throw new Error('Inference script not found');
        }

        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        const results = [];

        // Run analysis with each model
        for (const modelName of modelsToTest) {
            try {
                console.log(`Running analysis with ${modelName}...`);
                
                const result = await new Promise((resolve, reject) => {
                    const pythonProcess = spawn(pythonCmd, [
                        'inference.py', 
                        filePath, 
                        isVideo ? 'video' : 'image',
                        modelName
                    ], {
                        cwd: __dirname,
                        stdio: ['pipe', 'pipe', 'pipe']
                    });
                    
                    let stdout = '';
                    let stderr = '';
                    
                    pythonProcess.stdout.on('data', (data) => {
                        stdout += data.toString();
                    });
                    
                    pythonProcess.stderr.on('data', (data) => {
                        stderr += data.toString();
                    });
                    
                    pythonProcess.on('close', (code) => {
                        if (code === 0) {
                            resolve({ stdout: stdout.trim(), stderr });
                        } else {
                            reject(new Error(`Analysis failed for ${modelName} with code ${code}`));
                        }
                    });
                    
                    pythonProcess.on('error', (err) => {
                        reject(err);
                    });
                });

                // Parse result
                const lines = result.stdout.split('\n').filter(line => line.trim().length > 0);
                let jsonLine = lines.find(line => line.trim().startsWith('{'));
                
                if (jsonLine) {
                    const parsedResult = JSON.parse(jsonLine.trim());
                    results.push(parsedResult);
                    console.log(`${modelName}: ${parsedResult.prediction} (${(parsedResult.confidence * 100).toFixed(1)}%)`);
                } else {
                    console.warn(`No valid output from ${modelName}`);
                }

            } catch (error) {
                console.error(`Error with model ${modelName}:`, error.message);
                // Continue with other models even if one fails
                results.push({
                    model: modelName,
                    error: error.message,
                    status: 'failed'
                });
            }
        }

        // Clean up
        fs.removeSync(filePath);
        
        console.log(`=== MULTI-MODEL ANALYSIS END (${results.length} results) ===`);
        res.json({
            results,
            summary: {
                total_models: modelsToTest.length,
                successful: results.filter(r => r.status !== 'failed').length,
                failed: results.filter(r => r.status === 'failed').length
            },
            status: 'success'
        });

    } catch (error) {
        console.error('Multi-model analysis error:', error);
        
        // Clean up file
        if (req.file && req.file.path) {
            try {
                fs.removeSync(req.file.path);
            } catch (cleanupError) {
                console.error('Failed to cleanup file:', cleanupError);
            }
        }
        
        res.status(500).json({ 
            error: 'Multi-model analysis failed',
            details: error.message
        });
    }
});

// Get model comparison data
app.get('/api/comparison', (req, res) => {
    const models = Object.values(modelDatabase);
    
    const comparisonData = {
        models: models.map(model => ({
            name: model.name,
            validation: model.validation,
            testing: model.testing,
            accuracy: model.accuracy,
            downloads: model.downloads
        })),
        statistics: {
            average_validation: models.reduce((sum, m) => sum + m.validation, 0) / models.length,
            average_testing: models.reduce((sum, m) => sum + m.testing, 0) / models.length,
            best_model: models.reduce((best, current) => 
                current.testing > best.testing ? current : best
            ),
            total_models: models.length
        }
    };
    
    res.json(comparisonData);
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        message: 'Enhanced deepfake detector is running',
        models: Object.keys(modelDatabase).length,
        timestamp: new Date().toISOString()
    });
});

// Debug endpoint
app.get('/api/debug', (req, res) => {
    res.json({
        platform: process.platform,
        node_version: process.version,
        working_directory: process.cwd(),
        models_in_database: Object.keys(modelDatabase),
        inference_script_exists: fs.existsSync(path.join(__dirname, 'inference.py'))
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large. Maximum size is 50MB.' });
        }
    }
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Enhanced Deepfake Detector server running on port ${PORT}`);
    console.log(`📱 Access the app at: http://localhost:${PORT}`);
    console.log(`🧠 Active models: ${Object.keys(modelDatabase).length}`);
    console.log(`Platform: ${process.platform}`);
    console.log(`Node version: ${process.version}`);
});

module.exports = app;