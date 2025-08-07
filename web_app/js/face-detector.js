/**
 * Face Detection Module using ONNX Runtime
 * Handles model loading and inference for RetinaFace
 */

import { 
    handleError, 
    showLoading, 
    hideLoading,
    PerformanceMonitor
} from './utils.js';

export class FaceDetector {
    constructor() {
        this.session = null;
        this.modelLoaded = false;
        this.performanceMonitor = new PerformanceMonitor();
        
        // Model configuration
        this.config = {
            confidenceThreshold: 0.7,
            nmsThreshold: 0.4,
            inputSize: [640, 640]
        };
    }
    
    /**
     * Initialize ONNX Runtime and load the model
     */
    async initialize(modelPath = './models/optimized_model.onnx') {
        try {
            showLoading('Initializing ONNX Runtime...');
            
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded. Please ensure onnxruntime-web is included.');
            }
            
            // Configure ONNX Runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
            
            showLoading('Loading face detection model...');
            
            // Load model
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['webgl', 'wasm'],
                graphOptimizationLevel: 'all'
            });
            
            console.log('Model loaded successfully');
            console.log('Input names:', this.session.inputNames);
            console.log('Output names:', this.session.outputNames);
            
            this.modelLoaded = true;
            hideLoading();
            
            return true;
        } catch (error) {
            hideLoading();
            handleError(error, 'Model Loading');
            throw error;
        }
    }
    
    /**
     * Preprocess image data for model input
     */
    preprocessImage(imageData) {
        const { width: srcWidth, height: srcHeight } = imageData;
        const [targetHeight, targetWidth] = this.config.inputSize;
        
        // Create canvas for resizing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        
        // Calculate scale to maintain aspect ratio
        const scaleX = targetWidth / srcWidth;
        const scaleY = targetHeight / srcHeight;
        const scale = Math.min(scaleX, scaleY);
        
        const newWidth = srcWidth * scale;
        const newHeight = srcHeight * scale;
        const offsetX = (targetWidth - newWidth) / 2;
        const offsetY = (targetHeight - newHeight) / 2;
        
        // Fill with mean color
        ctx.fillStyle = 'rgb(104, 117, 123)';
        ctx.fillRect(0, 0, targetWidth, targetHeight);
        
        // Create temporary canvas for source image
        const srcCanvas = document.createElement('canvas');
        const srcCtx = srcCanvas.getContext('2d');
        srcCanvas.width = srcWidth;
        srcCanvas.height = srcHeight;
        srcCtx.putImageData(imageData, 0, 0);
        
        // Draw resized image
        ctx.drawImage(srcCanvas, offsetX, offsetY, newWidth, newHeight);
        
        // Get resized image data
        const resizedImageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
        const { data } = resizedImageData;
        
        // Convert to tensor format (CHW) and normalize
        const tensor = new Float32Array(3 * targetWidth * targetHeight);
        
        for (let i = 0; i < targetHeight; i++) {
            for (let j = 0; j < targetWidth; j++) {
                const pixelIndex = (i * targetWidth + j) * 4;
                const tensorIndex = i * targetWidth + j;
                
                // Convert RGB to BGR and subtract means
                const r = data[pixelIndex];
                const g = data[pixelIndex + 1];
                const b = data[pixelIndex + 2];
                
                // BGR order with mean subtraction
                tensor[tensorIndex] = b - 104;                              // B channel
                tensor[targetWidth * targetHeight + tensorIndex] = g - 117; // G channel  
                tensor[2 * targetWidth * targetHeight + tensorIndex] = r - 123; // R channel
            }
        }
        
        return {
            tensor,
            scale,
            offsetX,
            offsetY,
            originalSize: { width: srcWidth, height: srcHeight }
        };
    }
    
    /**
     * Detect faces in an image
     */
    async detect(imageData) {
        if (!this.modelLoaded || !this.session) {
            throw new Error('Model not loaded. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Preprocess image
            const { tensor, scale, offsetX, offsetY, originalSize } = this.preprocessImage(imageData);
            
            // Create input tensor
            const inputTensor = new ort.Tensor('float32', tensor, [1, 3, ...this.config.inputSize]);
            
            // Run inference
            const feeds = {};
            feeds[this.session.inputNames[0]] = inputTensor;
            
            const results = await this.session.run(feeds);
            
            // Extract outputs: [boxes, scores, landmarks]
            const boxes = results['output0'];  // [1, 16800, 4]
            const scores = results['593'];      // [1, 16800, 2] 
            const landmarks = results['592'];   // [1, 16800, 10]
            
            // Post-process results
            const detections = this.postProcessDetections(boxes, scores, landmarks, scale, offsetX, offsetY, originalSize);
            
            const inferenceTime = performance.now() - startTime;
            this.performanceMonitor.recordInference(inferenceTime);
            
            return {
                detections,
                inferenceTime,
                stats: this.performanceMonitor.getStats()
            };
            
        } catch (error) {
            handleError(error, 'Face Detection');
            throw error;
        }
    }
    
    /**
     * Post-process model outputs to get final detections
     */
    postProcessDetections(boxes, scores, landmarks, scale, offsetX, offsetY, originalSize) {
        const detections = [];
        const numDetections = boxes.dims[1]; // 16800
        
        // Get data arrays
        const boxData = boxes.data;
        const scoreData = scores.data;
        const landmarkData = landmarks.data;
        
        for (let i = 0; i < numDetections; i++) {
            // Get confidence score (face class, index 1)
            const confidence = scoreData[i * 2 + 1];
            
            if (confidence < this.config.confidenceThreshold) {
                continue;
            }
            
            // Get bounding box coordinates
            const x1 = boxData[i * 4];
            const y1 = boxData[i * 4 + 1];
            const x2 = boxData[i * 4 + 2];
            const y2 = boxData[i * 4 + 3];
            
            // Transform coordinates back to original image space
            const origX1 = Math.max(0, (x1 - offsetX) / scale);
            const origY1 = Math.max(0, (y1 - offsetY) / scale);
            const origX2 = Math.min(originalSize.width, (x2 - offsetX) / scale);
            const origY2 = Math.min(originalSize.height, (y2 - offsetY) / scale);
            
            // Get landmarks
            const detectionLandmarks = [];
            for (let j = 0; j < 5; j++) {
                const lx = (landmarkData[i * 10 + j * 2] - offsetX) / scale;
                const ly = (landmarkData[i * 10 + j * 2 + 1] - offsetY) / scale;
                detectionLandmarks.push(lx, ly);
            }
            
            detections.push({
                bbox: [origX1, origY1, origX2, origY2],
                confidence: confidence,
                landmarks: detectionLandmarks
            });
        }
        
        // Apply Non-Maximum Suppression
        return this.applyNMS(detections);
    }
    
    /**
     * Apply Non-Maximum Suppression
     */
    applyNMS(detections) {
        if (detections.length === 0) return [];
        
        // Sort by confidence
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const selected = [];
        const suppressed = new Set();
        
        for (let i = 0; i < detections.length; i++) {
            if (suppressed.has(i)) continue;
            
            const detection = detections[i];
            selected.push(detection);
            
            // Suppress overlapping detections
            for (let j = i + 1; j < detections.length; j++) {
                if (suppressed.has(j)) continue;
                
                const other = detections[j];
                const iou = this.calculateIoU(detection.bbox, other.bbox);
                
                if (iou > this.config.nmsThreshold) {
                    suppressed.add(j);
                }
            }
        }
        
        return selected;
    }
    
    /**
     * Calculate Intersection over Union (IoU)
     */
    calculateIoU(box1, box2) {
        const [x1_1, y1_1, x2_1, y2_1] = box1;
        const [x1_2, y1_2, x2_2, y2_2] = box2;
        
        const x1 = Math.max(x1_1, x1_2);
        const y1 = Math.max(y1_1, y1_2);
        const x2 = Math.min(x2_1, x2_2);
        const y2 = Math.min(y2_1, y2_2);
        
        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        const union = area1 + area2 - intersection;
        
        return union > 0 ? intersection / union : 0;
    }
    
    /**
     * Update model configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('Updated config:', this.config);
    }
    
    /**
     * Get current model configuration
     */
    getConfig() {
        return { ...this.config };
    }
    
    /**
     * Get model information
     */
    getModelInfo() {
        if (!this.session) {
            return null;
        }
        
        return {
            inputNames: this.session.inputNames,
            outputNames: this.session.outputNames,  
            inputSize: this.config.inputSize,
            loaded: this.modelLoaded
        };
    }
    
    /**
     * Cleanup resources
     */
    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        this.modelLoaded = false; 
        console.log('FaceDetector disposed');
    }
}

// Export singleton instance
export default new FaceDetector();
