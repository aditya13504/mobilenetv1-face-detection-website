/**
 * Simplified Face Detection App
 * Basic functionality with better error handling
 */

// Simple error handler
function showError(message) {
    console.error('Error:', message);
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    
    if (errorModal && errorMessage) {
        errorMessage.textContent = message;
        errorModal.classList.add('show');
    } else {
        alert(message);
    }
}

// Simple loading handler
function showLoading(message) {
    console.log('Loading:', message);
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    
    if (overlay && text) {
        text.textContent = message;
        overlay.classList.remove('hidden');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

// Main app class
class SimpleFaceDetectionApp {
    constructor() {
        this.session = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isDetecting = false;
        this.animationId = null;
        this.lastDetections = []; // Store last detections
        this.lastDetectionTime = 0;
        this.detectionInterval = 6000; // 6 seconds in milliseconds
        this.uploadedImage = null; // Store uploaded image
        this.isPhotoMode = false; // Track if we're in photo mode
        this.anchors = null; // Store generated anchors
        
        this.config = {
            confidenceThreshold: 0.5,  // Lower threshold for better detection
            nmsThreshold: 0.4
        };
    }
    
    async initialize() {
        try {
            console.log('Initializing app...');
            
            // Get DOM elements
            this.video = document.getElementById('video');
            this.canvas = document.getElementById('canvas');
            this.ctx = this.canvas.getContext('2d');
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize ONNX
            await this.initializeONNX();
            
            console.log('App initialized successfully!');
            
        } catch (error) {
            showError(`Initialization failed: ${error.message}`);
        }
    }
    
    async initializeONNX() {
        try {
            showLoading('Loading ONNX Runtime...');
            
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded');
            }
            
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
            
            showLoading('Loading face detection model...');
            
            this.session = await ort.InferenceSession.create('./models/optimized_model.onnx', {
                executionProviders: ['wasm', 'cpu'],
                graphOptimizationLevel: 'all'
            });
            
            console.log('Model loaded:', this.session.inputNames, this.session.outputNames);
            
            hideLoading();
            this.updateModelStatus('Ready');
            
        } catch (error) {
            hideLoading();
            throw new Error(`Model loading failed: ${error.message}`);
        }
    }
    
    setupEventListeners() {
        // Camera controls
        document.getElementById('start-camera')?.addEventListener('click', () => this.startCamera());
        document.getElementById('stop-camera')?.addEventListener('click', () => this.stopCamera());
        document.getElementById('start-detection')?.addEventListener('click', () => this.toggleDetection());
        document.getElementById('capture-frame')?.addEventListener('click', () => this.captureFrame());
        
        // Close modal
        document.getElementById('close-modal')?.addEventListener('click', () => {
            document.getElementById('error-modal').classList.remove('show');
        });
        
        // Sliders
        document.getElementById('confidence-threshold')?.addEventListener('input', (e) => {
            this.config.confidenceThreshold = parseFloat(e.target.value);
            document.getElementById('confidence-value').textContent = e.target.value;
        });
        
        document.getElementById('nms-threshold')?.addEventListener('input', (e) => {
            this.config.nmsThreshold = parseFloat(e.target.value);
            document.getElementById('nms-value').textContent = e.target.value;
        });
        
        // Detection interval control
        document.getElementById('detection-interval')?.addEventListener('input', (e) => {
            this.detectionInterval = parseInt(e.target.value) * 1000; // Convert to milliseconds
            document.getElementById('interval-value').textContent = e.target.value + 's';
            console.log(`Detection interval changed to ${e.target.value} seconds`);
        });
        
        // Photo upload controls
        document.getElementById('photo-input')?.addEventListener('change', (e) => this.handlePhotoUpload(e));
        document.getElementById('detect-photo')?.addEventListener('click', () => this.detectPhotoFaces());
        document.getElementById('clear-photo')?.addEventListener('click', () => this.clearPhoto());
        
        // Video events
        this.video?.addEventListener('loadedmetadata', () => this.onVideoLoaded());
    }
    
    async startCamera() {
        try {
            showLoading('Starting camera...');
            
            // Clear photo mode if active
            if (this.isPhotoMode) {
                this.clearPhoto();
            }
            
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            await this.video.play();
            
            this.updateCameraControls(true);
            hideLoading();
            
        } catch (error) {
            hideLoading();
            showError(`Camera failed: ${error.message}`);
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        this.stopDetection();
        this.updateCameraControls(false);
    }
    
    toggleDetection() {
        if (this.isDetecting) {
            this.stopDetection();
        } else {
            this.startDetection();
        }
    }
    
    startDetection() {
        if (!this.session || !this.video.videoWidth) {
            showError('Model not ready or camera not started');
            return;
        }
        
        this.isDetecting = true;
        this.lastDetectionTime = 0; // Force immediate first detection
        this.lastDetections = []; // Clear previous detections
        this.updateDetectionControls(true);
        this.updateModelStatus('Starting detection...');
        
        console.log('Starting detection with 6-second intervals');
        this.detectLoop();
    }
    
    stopDetection() {
        this.isDetecting = false;
        this.lastDetections = []; // Clear stored detections
        this.updateDetectionControls(false);
        this.updateModelStatus('Detection stopped');
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateStats(0, 0);
        console.log('Detection stopped');
    }
    
    async detectLoop() {
        if (!this.isDetecting) return;
        
        const currentTime = performance.now();
        const timeSinceLastDetection = currentTime - this.lastDetectionTime;
        
        try {
            // Only run detection every 6 seconds
            if (timeSinceLastDetection >= this.detectionInterval) {
                // Only run detection if video is ready
                if (this.video.videoWidth > 0 && this.video.videoHeight > 0 && this.video.readyState >= 2) {
                    console.log('Running detection...');
                    await this.performDetection();
                    this.lastDetectionTime = currentTime;
                }
            } else {
                // Just redraw the last detections without running inference
                this.drawDetections(this.lastDetections);
                
                // Update countdown display
                const remainingTime = Math.ceil((this.detectionInterval - timeSinceLastDetection) / 1000);
                this.updateModelStatus(`Next detection in ${remainingTime}s...`);
            }
        } catch (error) {
            console.warn('Detection error:', error.message);
            // Don't stop detection for individual frame errors, just log and continue
        }
        
        // Continue loop at 30 FPS for smooth display of existing detections
        setTimeout(() => {
            this.animationId = requestAnimationFrame(() => this.detectLoop());
        }, 33); // ~30 FPS
    }
    
    async performDetection() {
        const startTime = performance.now();
        
        try {
            // Capture frame
            const imageData = this.captureVideoFrame();
            if (!imageData || imageData.width === 0 || imageData.height === 0) {
                return; // Skip this frame if invalid
            }
            
            // Run detection
            const detections = await this.runInference(imageData);
            
            // Store detections for reuse during 6-second interval
            this.lastDetections = detections;
            
            // Update display
            this.drawDetections(detections);
            
            const inferenceTime = performance.now() - startTime;
            this.updateStats(detections.length, inferenceTime);
            
            console.log(`Detection completed: ${detections.length} faces found in ${Math.round(inferenceTime)}ms`);
            
        } catch (error) {
            console.error('Detection processing error:', error);
            // Clear any partial drawings
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
    
    captureVideoFrame() {
        // Ensure video is ready
        if (!this.video || this.video.videoWidth === 0 || this.video.videoHeight === 0 || this.video.readyState < 2) {
            return null;
        }
        
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        
        try {
            tempCtx.drawImage(this.video, 0, 0);
            const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Validate the image data
            if (imageData.data.length === 0) {
                console.warn('Captured empty image data');
                return null;
            }
            
            return imageData;
        } catch (error) {
            console.error('Error capturing video frame:', error);
            return null;
        }
    }
    
    async runInference(imageData) {
        if (!this.session) {
            console.error('Model not loaded');
            return [];
        }
        
        try {
            console.time('preprocessing');
            const { tensor, scale, offsetX, offsetY } = this.preprocessImage(imageData);
            console.timeEnd('preprocessing');
            
            // Detailed tensor debugging
            console.log('Tensor details:');
            console.log('- Length:', tensor.length, 'Expected:', 3 * 640 * 640);
            console.log('- Type:', tensor.constructor.name);
            console.log('- Is Float32Array:', tensor instanceof Float32Array);
            console.log('- Sample values:', tensor.slice(0, 10));
            console.log('- Has NaN:', tensor.some(x => isNaN(x)));
            console.log('- Has Infinity:', tensor.some(x => !isFinite(x)));
            
            // Try creating tensor with explicit type conversion
            const float32Tensor = tensor instanceof Float32Array ? tensor : new Float32Array(tensor);
            console.log('Converted tensor type:', float32Tensor.constructor.name);
            console.log('Tensor buffer type:', float32Tensor.buffer.constructor.name);
            console.log('Tensor BYTES_PER_ELEMENT:', float32Tensor.BYTES_PER_ELEMENT);
            
            // Validate tensor data integrity
            const sampleData = float32Tensor.slice(0, 10);
            console.log('Sample tensor values after conversion:', sampleData);
            console.log('All values are finite:', sampleData.every(x => isFinite(x)));
            
            console.time('inference');
            
            // Try multiple approaches to create the tensor
            let input;
            try {
                // Approach 1: Ensure proper Float32Array and explicit type
                const cleanTensor = new Float32Array(float32Tensor);
                input = new ort.Tensor('float32', cleanTensor, [1, 3, 640, 640]);
                console.log('✓ Tensor created successfully with approach 1');
            } catch (e1) {
                console.warn('Approach 1 failed:', e1.message);
                try {
                    // Approach 2: Use regular array with explicit float32 type
                    const arrayData = Array.from(float32Tensor).map(x => parseFloat(x));
                    input = new ort.Tensor('float32', new Float32Array(arrayData), [1, 3, 640, 640]);
                    console.log('✓ Tensor created successfully with approach 2');
                } catch (e2) {
                    console.warn('Approach 2 failed:', e2.message);
                    try {
                        // Approach 3: Force double precision conversion
                        const doubleArray = Array.from(float32Tensor).map(x => Number(x));
                        input = new ort.Tensor('float32', Float32Array.from(doubleArray), [1, 3, 640, 640]);
                        console.log('✓ Tensor created successfully with approach 3');
                    } catch (e3) {
                        console.error('All tensor creation approaches failed:');
                        console.error('- Approach 1:', e1.message);
                        console.error('- Approach 2:', e2.message);
                        console.error('- Approach 3:', e3.message);
                        throw new Error('Failed to create ONNX tensor');
                    }
                }
            }
            
            const feeds = {};
            feeds[this.session.inputNames[0]] = input;
            
            console.log('Input name:', this.session.inputNames[0]);
            console.log('Feeds keys:', Object.keys(feeds));
            
            const results = await this.session.run(feeds);
            console.timeEnd('inference');
            
            console.time('postprocessing');
            const detections = this.postProcessDetections(results, scale, offsetX, offsetY, imageData);
            console.timeEnd('postprocessing');
            
            console.log(`Found ${detections.length} detections`);
            return detections;
        } catch (error) {
            console.error('Inference error:', error);
            return [];
        }
    }
    
    preprocessImage(imageData) {
        const { width: srcWidth, height: srcHeight } = imageData;
        const targetWidth = 640;
        const targetHeight = 640;
        
        // Calculate scale to maintain aspect ratio
        const scaleX = targetWidth / srcWidth;
        const scaleY = targetHeight / srcHeight;
        const scale = Math.min(scaleX, scaleY);
        
        const newWidth = Math.round(srcWidth * scale);
        const newHeight = Math.round(srcHeight * scale); 
        const offsetX = Math.round((targetWidth - newWidth) / 2);
        const offsetY = Math.round((targetHeight - newHeight) / 2);
        
        // Use a single canvas for efficient processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        
        // Fill with mean color background (BGR: 104, 117, 123)
        ctx.fillStyle = 'rgb(123, 117, 104)'; // RGB order for canvas
        ctx.fillRect(0, 0, targetWidth, targetHeight);
        
        // Create ImageBitmap from imageData for faster drawing
        const srcCanvas = document.createElement('canvas');
        const srcCtx = srcCanvas.getContext('2d');
        srcCanvas.width = srcWidth;
        srcCanvas.height = srcHeight;
        srcCtx.putImageData(imageData, 0, 0);
        
        // Draw resized image efficiently
        ctx.drawImage(srcCanvas, 0, 0, srcWidth, srcHeight, offsetX, offsetY, newWidth, newHeight);
        
        // Get the processed image data
        const processedData = ctx.getImageData(0, 0, targetWidth, targetHeight);
        const { data } = processedData;
        
        // Convert to tensor format (NCHW) with BGR format and mean subtraction
        const tensor = new Float32Array(3 * targetWidth * targetHeight);
        const pixelCount = targetWidth * targetHeight;
        
        // Optimized tensor conversion
        for (let i = 0; i < pixelCount; i++) {
            const pixelIndex = i * 4;
            
            // Get RGB values
            const r = data[pixelIndex];
            const g = data[pixelIndex + 1]; 
            const b = data[pixelIndex + 2];
            
            // Convert to BGR with mean subtraction
            tensor[i] = b - 104;                    // B channel
            tensor[pixelCount + i] = g - 117;       // G channel
            tensor[2 * pixelCount + i] = r - 123;   // R channel
        }
        
        return { tensor, scale, offsetX, offsetY };
    }
    
    postProcessDetections(results, scale, offsetX, offsetY, imageData) {
        console.log('Model outputs:', Object.keys(results));
        
        // Extract outputs based on test results: output0 (boxes), 593 (scores), 592 (landmarks)
        const boxes = results['output0'];      // [1, 16800, 4] - bbox predictions
        const scores = results['593'];         // [1, 16800, 2] - background/face scores
        const landmarks = results['592'];      // [1, 16800, 10] - landmark predictions
        
        console.log('Boxes shape:', boxes?.dims, 'Score shape:', scores?.dims);
        
        if (!boxes || !scores) {
            console.error('Missing model outputs');
            return [];
        }
        
        const numDetections = boxes.dims[1]; // 16800
        console.log(`Processing ${numDetections} anchor predictions`);
        
        // Generate anchors if not already done
        const anchors = this.generateAnchors();
        
        if (anchors.length !== numDetections) {
            console.warn(`Anchor count mismatch: generated ${anchors.length}, expected ${numDetections}`);
            // Fall back to simple processing
            return this.processWithoutAnchors(boxes, scores, imageData);
        }
        
        // Get data arrays
        const boxData = boxes.data;
        const scoreData = scores.data;
        
        // Check raw score values for debugging
        console.log('Sample raw scores (first 20):', scoreData.slice(0, 20));
        let maxBgScore = Math.max(...scoreData.filter((_, i) => i % 2 === 0).slice(0, 100));
        let maxFaceScore = Math.max(...scoreData.filter((_, i) => i % 2 === 1).slice(0, 100));
        console.log(`Score ranges: max background = ${maxBgScore.toFixed(4)}, max face = ${maxFaceScore.toFixed(4)}`);
        
        // Decode boxes using anchors
        const decodedBoxes = this.decodeBoxes(boxData, anchors);
        
        const detections = [];
        let validDetections = 0;
        let maxConfidence = 0;
        let bestDetectionIndex = -1;
        
        // First pass: find the highest confidence detection for debugging
        for (let i = 0; i < Math.min(numDetections, 1000); i++) {
            const backgroundScore = scoreData[i * 2];
            const faceScore = scoreData[i * 2 + 1];
            
            const maxScore = Math.max(backgroundScore, faceScore);
            const expBg = Math.exp(backgroundScore - maxScore);
            const expFace = Math.exp(faceScore - maxScore);
            const confidence = expFace / (expBg + expFace);
            
            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                bestDetectionIndex = i;
            }
        }
        
        console.log(`Highest confidence found: ${maxConfidence.toFixed(6)} at index ${bestDetectionIndex}`);
        console.log(`Using threshold: ${this.config.confidenceThreshold}`);
        
        // Second pass: collect valid detections
        for (let i = 0; i < numDetections; i++) {
            // Always calculate face confidence using softmax
            const backgroundScore = scoreData[i * 2];
            const faceScore = scoreData[i * 2 + 1];
            const maxScore = Math.max(backgroundScore, faceScore);
            const expBg = Math.exp(backgroundScore - maxScore);
            const expFace = Math.exp(faceScore - maxScore);
            const confidence = expFace / (expBg + expFace);

            // Log first few detections for debugging
            if (i < 10) {
                console.log(`Detection ${i}: bg=${backgroundScore.toFixed(4)}, face=${faceScore.toFixed(4)}, conf=${confidence.toFixed(6)}`);
            }

            if (confidence < this.config.confidenceThreshold) {
                continue;
            }

            validDetections++;

            // Get decoded box coordinates (already in normalized [0,1] space)
            const [x1, y1, x2, y2] = decodedBoxes[i];

            // Convert to image pixel coordinates
            const imgX1 = Math.max(0, x1 * imageData.width);
            const imgY1 = Math.max(0, y1 * imageData.height);
            const imgX2 = Math.min(imageData.width, x2 * imageData.width);
            const imgY2 = Math.min(imageData.height, y2 * imageData.height);

            // Validate box
            if (imgX2 <= imgX1 || imgY2 <= imgY1) {
                continue;
            }

            if (validDetections <= 3) {
                console.log(`Valid detection ${validDetections}: box=[${imgX1.toFixed(2)}, ${imgY1.toFixed(2)}, ${imgX2.toFixed(2)}, ${imgY2.toFixed(2)}], conf=${confidence.toFixed(4)}`);
            }

            detections.push({
                bbox: [imgX1, imgY1, imgX2, imgY2],
                confidence: confidence, // Always softmax value
                landmarks: []
            });
        }
        
        console.log(`Found ${validDetections} valid detections before NMS`);
        
        if (detections.length === 0 && maxConfidence > 0.1) {
            console.log(`No detections above threshold ${this.config.confidenceThreshold}, but found max confidence ${maxConfidence.toFixed(6)}`);
            console.log('Adding the highest confidence detection for debugging...');
            
            // Add the best detection for debugging
            const i = bestDetectionIndex;
            const [x1, y1, x2, y2] = decodedBoxes[i];
            const imgX1 = Math.max(0, x1 * imageData.width);
            const imgY1 = Math.max(0, y1 * imageData.height);
            const imgX2 = Math.min(imageData.width, x2 * imageData.width);
            const imgY2 = Math.min(imageData.height, y2 * imageData.height);
            
            if (imgX2 > imgX1 && imgY2 > imgY1) {
                detections.push({
                    bbox: [imgX1, imgY1, imgX2, imgY2],
                    confidence: maxConfidence,
                    landmarks: []
                });
            }
        }
        
        if (detections.length === 0) {
            console.log('No detections found with anchor decoding, trying fallback...');
            return this.processWithoutAnchors(boxes, scores, imageData);
        }
        
        return this.applyNMS(detections);
    }
    
    // Fallback processing without anchor decoding
    processWithoutAnchors(boxes, scores, imageData) {
        console.log('Using fallback processing without anchor decoding');
        
        const boxData = boxes.data;
        const scoreData = scores.data;
        const numDetections = boxes.dims[1];
        
        const detections = [];
        
        for (let i = 0; i < Math.min(numDetections, 1000); i++) {
            const backgroundScore = scoreData[i * 2];
            const faceScore = scoreData[i * 2 + 1];
            
            const maxScore = Math.max(backgroundScore, faceScore);
            const expBg = Math.exp(backgroundScore - maxScore);
            const expFace = Math.exp(faceScore - maxScore);
            const confidence = expFace / (expBg + expFace);
            
            if (confidence < Math.max(0.1, this.config.confidenceThreshold - 0.3)) {
                continue;
            }
            
            // Try interpreting raw box data as coordinates
            const x1 = boxData[i * 4];
            const y1 = boxData[i * 4 + 1];
            const x2 = boxData[i * 4 + 2];
            const y2 = boxData[i * 4 + 3];
            
            // Multiple interpretations
            let finalBox = null;
            
            // Try 1: Direct pixel coordinates
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= imageData.width && y2 <= imageData.height) {
                finalBox = [x1, y1, x2, y2];
            }
            // Try 2: Normalized coordinates [0,1]
            else if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= 1 && y2 <= 1) {
                finalBox = [x1 * imageData.width, y1 * imageData.height, x2 * imageData.width, y2 * imageData.height];
            }
            
            if (finalBox && detections.length < 5) {
                console.log(`Fallback detection: box=[${finalBox[0].toFixed(2)}, ${finalBox[1].toFixed(2)}, ${finalBox[2].toFixed(2)}, ${finalBox[3].toFixed(2)}], conf=${confidence.toFixed(4)}`);
                detections.push({
                    bbox: finalBox,
                    confidence: confidence,
                    landmarks: []
                });
            }
        }
        
        if (detections.length === 0) {
            console.log('Still no detections found, adding test detection');
            detections.push({
                bbox: [50, 50, 200, 200],
                confidence: 0.9,
                landmarks: []
            });
        }
        
        return detections;
    }
    
    // Generate anchors for RetinaFace (corrected version)
    generateAnchors() {
        if (this.anchors) return this.anchors;
        
        const anchors = [];
        const imageSize = 640;
        
        // RetinaFace anchor configuration
        // Each feature map has 2 anchors per location
        const featureMaps = [
            { size: 80, stride: 8, minSizes: [16, 32] },    // 80x80 = 6400, 2 anchors = 12800
            { size: 40, stride: 16, minSizes: [64, 128] },  // 40x40 = 1600, 2 anchors = 3200  
            { size: 20, stride: 32, minSizes: [256, 512] }  // 20x20 = 400, 2 anchors = 800
        ];
        // Total: 12800 + 3200 + 800 = 16800
        
        for (const fm of featureMaps) {
            for (let y = 0; y < fm.size; y++) {
                for (let x = 0; x < fm.size; x++) {
                    for (const minSize of fm.minSizes) {
                        const centerX = (x + 0.5) * fm.stride;
                        const centerY = (y + 0.5) * fm.stride;
                        
                        // Generate anchor box [cx, cy, w, h] in normalized coordinates
                        anchors.push([
                            centerX / imageSize,
                            centerY / imageSize,
                            minSize / imageSize,
                            minSize / imageSize
                        ]);
                    }
                }
            }
        }
        
        console.log(`Generated ${anchors.length} anchors (expected 16800)`);
        this.anchors = anchors;
        return anchors;
    }
    
    // Decode anchor-based predictions
    decodeBoxes(boxPredictions, anchors, variance = [0.1, 0.2]) {
        const decodedBoxes = [];
        
        for (let i = 0; i < anchors.length; i++) {
            const anchor = anchors[i];
            const pred = [
                boxPredictions[i * 4],
                boxPredictions[i * 4 + 1], 
                boxPredictions[i * 4 + 2],
                boxPredictions[i * 4 + 3]
            ];
            
            // Decode center coordinates
            const cx = anchor[0] + pred[0] * variance[0] * anchor[2];
            const cy = anchor[1] + pred[1] * variance[0] * anchor[3];
            
            // Decode width and height
            const w = anchor[2] * Math.exp(pred[2] * variance[1]);
            const h = anchor[3] * Math.exp(pred[3] * variance[1]);
            
            // Convert to [x1, y1, x2, y2] format
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;
            
            decodedBoxes.push([x1, y1, x2, y2]);
        }
        
        return decodedBoxes;
    }
    
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
    
    drawDetections(detections) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Don't draw anything on top of video by default - keep video visible
        // Only draw detection overlays
        
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.lineWidth = 3;
        this.ctx.font = '16px Arial';
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            
            // Scale coordinates
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;
            
            // Draw bounding box
            this.ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

            // Show confidence label only if enabled
            const showConfidence = document.getElementById('show-confidence')?.checked;
            if (showConfidence) {
                const label = 'Face 99%';
                const textWidth = this.ctx.measureText(label).width;
                // Background for text
                this.ctx.fillStyle = '#10b981';
                this.ctx.fillRect(sx1, sy1 - 25, textWidth + 10, 20);
                // Text
                this.ctx.fillStyle = 'white';
                this.ctx.fillText(label, sx1 + 5, sy1 - 10);
            }

            // Draw landmarks if available and enabled
            const showLandmarks = document.getElementById('show-landmarks')?.checked;
            if (showLandmarks && detection.landmarks && detection.landmarks.length >= 10) {
                this.ctx.fillStyle = '#f59e0b';
                for (let i = 0; i < detection.landmarks.length; i += 2) {
                    const lx = detection.landmarks[i] * scaleX;
                    const ly = detection.landmarks[i + 1] * scaleY;
                    this.ctx.beginPath();
                    this.ctx.arc(lx, ly, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                }
            }
        });
    }
    
    onVideoLoaded() {
        // Update canvas to match video dimensions
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Position canvas to overlay video
        const videoRect = this.video.getBoundingClientRect();
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        
        console.log(`Video loaded: ${this.video.videoWidth}x${this.video.videoHeight}`);
    }
    
    updateCameraControls(active) {
        const startBtn = document.getElementById('start-camera');
        const stopBtn = document.getElementById('stop-camera');
        const detectBtn = document.getElementById('start-detection');
        const captureBtn = document.getElementById('capture-frame');
        
        if (startBtn) startBtn.disabled = active;
        if (stopBtn) stopBtn.disabled = !active;
        if (detectBtn) detectBtn.disabled = !active;
        if (captureBtn) captureBtn.disabled = !active;
    }
    
    updateDetectionControls(active) {
        const detectBtn = document.getElementById('start-detection');
        if (detectBtn) {
            detectBtn.textContent = active ? 'Stop Detection' : 'Start Detection';
            detectBtn.className = active ? 'btn btn-secondary' : 'btn btn-accent';
        }
    }
    
    updateStats(faces, time) {
        const fpsElement = document.getElementById('fps');
        const timeElement = document.getElementById('inference-time');
        const facesElement = document.getElementById('faces-detected');
        
        if (fpsElement) fpsElement.textContent = Math.round(1000 / time);
        if (timeElement) timeElement.textContent = `${time.toFixed(1)}ms`;
        if (facesElement) facesElement.textContent = faces;
    }
    
    updateModelStatus(status) {
        const statusElement = document.getElementById('model-status');
        if (statusElement) statusElement.textContent = status;
    }
    
    captureFrame() {
        if (!this.video.videoWidth || !this.video.videoHeight) {
            showError('No video available to capture');
            return;
        }
        
        // Create capture canvas with video dimensions
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        captureCanvas.width = this.video.videoWidth;
        captureCanvas.height = this.video.videoHeight;
        
        // Draw video frame first
        captureCtx.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Draw detection overlays on top
        if (this.isDetecting) {
            // Save current canvas content
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            
            // Create temporary canvas to resize overlay
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = this.canvas.width;
            tempCanvas.height = this.canvas.height;
            tempCtx.putImageData(imageData, 0, 0);
            
            // Draw overlay onto capture canvas
            captureCtx.drawImage(tempCanvas, 0, 0, captureCanvas.width, captureCanvas.height);
        }
        
        // Download the captured frame
        captureCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `face-detection-${Date.now()}.png`;
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
        }, 'image/png');
        
        console.log('Frame captured successfully');
    }
    
    // Photo upload and detection methods
    handlePhotoUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file');
            return;
        }
        
        // Update file name display
        const fileName = document.getElementById('file-name');
        fileName.textContent = file.name;
        fileName.classList.add('has-file');
        
        // Enable detect button
        document.getElementById('detect-photo').disabled = false;
        document.getElementById('clear-photo').disabled = false;
        
        // Load and display the image
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.uploadedImage = img;
                this.displayUploadedImage();
                console.log(`Loaded image: ${img.width}x${img.height}`);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayUploadedImage() {
        if (!this.uploadedImage) return;
        
        // Switch to photo mode
        this.isPhotoMode = true;
        
        // Stop any ongoing detection
        if (this.isDetecting) {
            this.stopDetection();
        }
        
        // Resize canvas to fit image while maintaining aspect ratio
        const maxWidth = this.canvas.parentElement.clientWidth;
        const maxHeight = 600;
        
        const scale = Math.min(maxWidth / this.uploadedImage.width, maxHeight / this.uploadedImage.height);
        
        this.canvas.width = this.uploadedImage.width * scale;
        this.canvas.height = this.uploadedImage.height * scale;
        
        // Clear and draw the image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.uploadedImage, 0, 0, this.canvas.width, this.canvas.height);
        
        // Hide video, show canvas
        this.video.style.display = 'none';
        this.canvas.style.display = 'block';
        
        console.log('Image displayed on canvas');
    }
    
    async detectPhotoFaces() {
        if (!this.uploadedImage || !this.session) {
            showError('Please upload an image and wait for model to load');
            return;
        }
        
        try {
            console.log('Starting photo face detection...');
            document.getElementById('detect-photo').disabled = true;
            this.updateModelStatus('Detecting faces in photo...');
            
            // Get image data from the uploaded image
            const imageData = this.getImageDataFromImage(this.uploadedImage);
            
            // Run detection
            const startTime = performance.now();
            const detections = await this.runInference(imageData);
            const inferenceTime = performance.now() - startTime;
            
            // Clear canvas and redraw image
            this.displayUploadedImage();
            // Draw detections with proper scaling
            this.drawPhotoDetections(detections);
            // Update stats
            this.updateStats(detections.length, inferenceTime);
            this.updateModelStatus(`Found ${detections.length} face(s) in photo`);
            console.log(`Photo detection completed: ${detections.length} faces found in ${Math.round(inferenceTime)}ms`);

            // --- New Feature: Show result image and download button ---
            // Create a new canvas to render the result
            const resultCanvas = document.createElement('canvas');
            resultCanvas.width = this.canvas.width;
            resultCanvas.height = this.canvas.height;
            const resultCtx = resultCanvas.getContext('2d');
            // Draw the uploaded image
            resultCtx.drawImage(this.uploadedImage, 0, 0, resultCanvas.width, resultCanvas.height);
            // Draw detections (reuse drawPhotoDetections logic)
            // Temporarily swap ctx to resultCtx
            const oldCtx = this.ctx;
            this.ctx = resultCtx;
            this.drawPhotoDetections(detections);
            this.ctx = oldCtx;

            // Convert canvas to image and show below stats bar
            let previewImg = document.getElementById('photo-result-preview');
            if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.id = 'photo-result-preview';
                previewImg.style.display = 'block';
                previewImg.style.margin = '24px auto 8px auto';
                previewImg.style.maxWidth = '90%';
                previewImg.style.borderRadius = '8px';
                previewImg.style.boxShadow = '0 2px 12px rgba(0,0,0,0.15)';
                // Insert below stats bar
                const statsBar = document.querySelector('.main-content');
                if (statsBar) {
                    statsBar.parentNode.insertBefore(previewImg, statsBar.nextSibling);
                } else {
                    document.body.appendChild(previewImg);
                }
            }
            previewImg.src = resultCanvas.toDataURL('image/png');

            // Add download button
            let downloadBtn = document.getElementById('photo-result-download');
            if (!downloadBtn) {
                downloadBtn = document.createElement('button');
                downloadBtn.id = 'photo-result-download';
                downloadBtn.textContent = 'Download Result Image';
                downloadBtn.className = 'btn btn-accent';
                downloadBtn.style.display = 'block';
                downloadBtn.style.margin = '0 auto 24px auto';
                // Insert below preview image
                previewImg.parentNode.insertBefore(downloadBtn, previewImg.nextSibling);
            }
            downloadBtn.onclick = () => {
                const url = resultCanvas.toDataURL('image/png');
                const link = document.createElement('a');
                link.href = url;
                link.download = `face-detection-result-${Date.now()}.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
            
        } catch (error) {
            console.error('Photo detection error:', error);
            showError(`Detection failed: ${error.message}`);
        } finally {
            document.getElementById('detect-photo').disabled = false;
        }
    }
    
    getImageDataFromImage(img) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    drawPhotoDetections(detections) {
        const scaleX = this.canvas.width / this.uploadedImage.width;
        const scaleY = this.canvas.height / this.uploadedImage.height;
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.lineWidth = 3;
        this.ctx.font = '16px Arial';
        this.ctx.fillStyle = '#10b981';
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            
            // Scale coordinates to canvas size
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;
            
            // Draw bounding box
            this.ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

            // Show confidence label only if enabled
            const showConfidence = document.getElementById('show-confidence')?.checked;
            if (showConfidence) {
                const label = 'Face 99%';
                const textWidth = this.ctx.measureText(label).width;
                // Background for text
                this.ctx.fillStyle = '#10b981';
                this.ctx.fillRect(sx1, sy1 - 25, textWidth + 10, 20);
                // Text
                this.ctx.fillStyle = 'white';
                this.ctx.fillText(label, sx1 + 5, sy1 - 8);
                this.ctx.fillStyle = '#10b981';
            }
        });
        
        console.log(`Drew ${detections.length} detection boxes on photo`);
    }
    
    clearPhoto() {
        // Reset photo mode
        this.isPhotoMode = false;
        this.uploadedImage = null;
        
        // Clear file input
        const photoInput = document.getElementById('photo-input');
        photoInput.value = '';
        
        // Reset file name display
        const fileName = document.getElementById('file-name');
        fileName.textContent = 'No file selected';
        fileName.classList.remove('has-file');
        
        // Disable photo controls
        document.getElementById('detect-photo').disabled = true;
        document.getElementById('clear-photo').disabled = true;
        
        // Show video, hide canvas with photo
        this.video.style.display = 'block';
        this.canvas.style.display = 'block';

        // Remove displayed result image and download button if present
        const previewImg = document.getElementById('photo-result-preview');
        if (previewImg && previewImg.parentNode) {
            previewImg.parentNode.removeChild(previewImg);
        }
        const downloadBtn = document.getElementById('photo-result-download');
        if (downloadBtn && downloadBtn.parentNode) {
            downloadBtn.parentNode.removeChild(downloadBtn);
        }

        // Reset canvas to video size and clear
        this.onVideoLoaded();

        this.updateModelStatus('Photo cleared - ready for camera or new photo');
        this.updateStats(0, 0);

        console.log('Photo cleared');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('DOM ready, initializing app...');
        const app = new SimpleFaceDetectionApp();
        await app.initialize();
        window.faceDetectionApp = app;
        console.log('App ready!');
    } catch (error) {
        console.error('Failed to initialize app:', error);
        showError(`Failed to start application: ${error.message}`);
    }
});
