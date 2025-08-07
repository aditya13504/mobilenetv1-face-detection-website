/**
 * Utility functions for face detection web application
 */

/**
 * Download a file from a URL with progress tracking
 * @param {string} url - URL to download from
 * @param {Function} onProgress - Progress callback function
 * @returns {Promise<ArrayBuffer>} - Promise resolving to ArrayBuffer
 */
export async function downloadWithProgress(url, onProgress) {
    const response = await fetch(url);
    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength, 10);
    
    if (!response.ok) {
        throw new Error(`Failed to download ${url}: ${response.statusText}`);
    }
    
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    
    while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        chunks.push(value);
        loaded += value.length;
        
        if (onProgress && total) {
            onProgress(loaded / total);
        }
    }
    
    // Combine chunks into single ArrayBuffer
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    
    return result.buffer;
}

/**
 * Debounce function to limit function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Performance monitor for tracking FPS and inference times
 */
export class PerformanceMonitor {
    constructor() {
        this.frameTimes = [];
        this.inferenceTimes = [];
        this.lastFrameTime = 0;
        this.maxSamples = 30; // Keep last 30 samples
    }
    
    recordFrame() {
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const frameTime = now - this.lastFrameTime;
            this.frameTimes.push(frameTime);
            
            if (this.frameTimes.length > this.maxSamples) {
                this.frameTimes.shift();
            }
        }
        this.lastFrameTime = now;
    }
    
    recordInference(time) {
        this.inferenceTimes.push(time);
        if (this.inferenceTimes.length > this.maxSamples) {
            this.inferenceTimes.shift();
        }
    }
    
    getFPS() {
        if (this.frameTimes.length === 0) return 0;
        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        return Math.round(1000 / avgFrameTime);
    }
    
    getAvgInferenceTime() {
        if (this.inferenceTimes.length === 0) return 0;
        const avg = this.inferenceTimes.reduce((a, b) => a + b, 0) / this.inferenceTimes.length;
        return Math.round(avg * 100) / 100; // Round to 2 decimal places
    }
    
    getStats() {
        return {
            fps: this.getFPS(),
            avgInferenceTime: this.getAvgInferenceTime(),
            samples: this.inferenceTimes.length
        };
    }
}

/**
 * Format file size in human readable format
 * @param {number} bytes - Size in bytes
 * @returns {string} - Formatted string
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format time in milliseconds to human readable format
 * @param {number} ms - Time in milliseconds
 * @returns {string} - Formatted string
 */
export function formatTime(ms) {
    if (ms < 1000) {
        return `${Math.round(ms)}ms`;
    } else if (ms < 60000) {
        return `${(ms / 1000).toFixed(1)}s`;
    } else {
        const minutes = Math.floor(ms / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        return `${minutes}m ${seconds}s`;
    }
}

/**
 * Check if browser supports required features
 * @returns {Object} - Support status for various features
 */
export function checkBrowserSupport() {
    const support = {
        webgl: false,
        webgl2: false,
        getUserMedia: false,
        webAssembly: false,
        onnxjs: false
    };
    
    // Check WebGL support
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        support.webgl = !!gl;
        
        const gl2 = canvas.getContext('webgl2');
        support.webgl2 = !!gl2;
    } catch (e) {
        support.webgl = false;
        support.webgl2 = false;
    }
    
    // Check getUserMedia support
    support.getUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    
    // Check WebAssembly support
    support.webAssembly = typeof WebAssembly === 'object';
    
    // Check ONNX.js availability (will be checked after loading)
    support.onnxjs = typeof ort !== 'undefined';
    
    return support;
}

/**
 * Get camera constraints based on preferred resolution
 * @param {string} resolution - Preferred resolution ('720p', '480p', etc.)
 * @returns {Object} - MediaStream constraints
 */
export function getCameraConstraints(resolution = '720p') {
    const constraints = {
        video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
        },
        audio: false
    };
    
    switch (resolution) {
        case '1080p':
            constraints.video.width.ideal = 1920;
            constraints.video.height.ideal = 1080;
            break;
        case '720p':
            constraints.video.width.ideal = 1280;
            constraints.video.height.ideal = 720;
            break;
        case '480p':
            constraints.video.width.ideal = 640;
            constraints.video.height.ideal = 480;
            break;
        case '360p':
            constraints.video.width.ideal = 480;
            constraints.video.height.ideal = 360;
            break;
        default:
            // Use default 640x480
            break;
    }
    
    return constraints;
}

/**
 * Resize image data while maintaining aspect ratio
 * @param {ImageData} imageData - Source image data
 * @param {number} targetWidth - Target width
 * @param {number} targetHeight - Target height
 * @returns {Object} - Resized image data and scale factors
 */
export function resizeImageData(imageData, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const { width: srcWidth, height: srcHeight } = imageData;
    
    // Calculate scale factors
    const scaleX = targetWidth / srcWidth;
    const scaleY = targetHeight / srcHeight;
    const scale = Math.min(scaleX, scaleY);
    
    const newWidth = Math.round(srcWidth * scale);
    const newHeight = Math.round(srcHeight * scale);
    
    // Create temporary canvas for source image
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d');
    srcCanvas.width = srcWidth;
    srcCanvas.height = srcHeight;
    srcCtx.putImageData(imageData, 0, 0);
    
    // Resize to target canvas
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Clear with black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, targetWidth, targetHeight);
    
    // Draw resized image centered
    const offsetX = (targetWidth - newWidth) / 2;
    const offsetY = (targetHeight - newHeight) / 2;
    
    ctx.drawImage(srcCanvas, offsetX, offsetY, newWidth, newHeight);
    
    return {
        imageData: ctx.getImageData(0, 0, targetWidth, targetHeight),
        scaleX: scale,
        scaleY: scale,
        offsetX,
        offsetY
    };
}

/**
 * Convert image data to tensor format expected by ONNX model
 * @param {ImageData} imageData - Source image data
 * @returns {Float32Array} - Tensor data in CHW format
 */
export function imageDataToTensor(imageData) {
    const { data, width, height } = imageData;
    const tensor = new Float32Array(3 * width * height);
    
    // Convert RGBA to RGB and normalize to [0, 1]
    // Reorder from HWC to CHW format
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            const pixelIndex = (i * width + j) * 4;
            const tensorIndex = i * width + j;
            
            // R channel
            tensor[tensorIndex] = data[pixelIndex] / 255.0;
            // G channel  
            tensor[width * height + tensorIndex] = data[pixelIndex + 1] / 255.0;
            // B channel
            tensor[2 * width * height + tensorIndex] = data[pixelIndex + 2] / 255.0;
        }
    }
    
    return tensor;
}

/**
 * Apply Non-Maximum Suppression to detection results
 * @param {Array} detections - Array of detection objects
 * @param {number} threshold - NMS threshold
 * @returns {Array} - Filtered detections
 */
export function applyNMS(detections, threshold = 0.4) {
    if (detections.length === 0) return [];
    
    // Sort by confidence score (descending)
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
            const iou = calculateIoU(detection.bbox, other.bbox);
            
            if (iou > threshold) {
                suppressed.add(j);
            }
        }
    }
    
    return selected;
}

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes
 * @param {Array} box1 - [x1, y1, x2, y2]
 * @param {Array} box2 - [x1, y1, x2, y2]
 * @returns {number} - IoU value
 */
export function calculateIoU(box1, box2) {
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;
    
    // Calculate intersection area
    const x1 = Math.max(x1_1, x1_2);
    const y1 = Math.max(y1_1, y1_2);
    const x2 = Math.min(x2_1, x2_2);
    const y2 = Math.min(y2_1, y2_2);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    
    // Calculate union area
    const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    const union = area1 + area2 - intersection;
    
    return union > 0 ? intersection / union : 0;
}

/**
 * Error handler for displaying user-friendly error messages
 * @param {Error} error - Error object
 * @param {string} context - Context where error occurred
 */
export function handleError(error, context = 'Unknown') {
    console.error(`[${context}] Error:`, error);
    
    let userMessage = 'An unexpected error occurred.';
    
    if (error.name === 'NotAllowedError') {
        userMessage = 'Camera access was denied. Please allow camera permissions and refresh the page.';
    } else if (error.name === 'NotFoundError') {
        userMessage = 'No camera was found. Please ensure a camera is connected.';
    } else if (error.name === 'NetworkError') {
        userMessage = 'Failed to load model. Please check your internet connection.';
    } else if (error.message.includes('WebGL')) {
        userMessage = 'WebGL is not supported or disabled. Please enable hardware acceleration.';
    } else if (error.message.includes('ONNX')) {
        userMessage = 'Failed to initialize the AI model. Please refresh the page and try again.';
    }
    
    // Show error modal if available
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    
    if (errorModal && errorMessage) {
        errorMessage.textContent = userMessage;
        errorModal.classList.add('show');
    } else {
        alert(userMessage);
    }
}

/**
 * Show loading overlay with progress
 * @param {string} message - Loading message
 * @param {number} progress - Progress percentage (0-1)
 */
export function showLoading(message, progress = null) {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    
    if (overlay && text) {
        text.textContent = message;
        if (progress !== null && progress >= 0 && progress <= 1) {
            text.textContent += ` (${Math.round(progress * 100)}%)`;
        }
        overlay.classList.remove('hidden');
    }
}

/**
 * Hide loading overlay
 */
export function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

/**
 * Update statistics display
 * @param {Object} stats - Statistics object
 */
export function updateStats(stats) {
    const elements = {
        fps: document.getElementById('fps'),
        inferenceTime: document.getElementById('inference-time'),
        facesDetected: document.getElementById('faces-detected'),
        modelStatus: document.getElementById('model-status')
    };
    
    for (const [key, element] of Object.entries(elements)) {
        if (element && stats[key] !== undefined) {
            element.textContent = stats[key];
        }
    }
}

/**
 * Validate model input dimensions
 * @param {number} width - Input width
 * @param {number} height - Input height
 * @returns {boolean} - Whether dimensions are valid
 */
export function validateModelInput(width, height) {
    // RetinaFace typically expects certain input sizes
    const validSizes = [
        { width: 640, height: 640 },
        { width: 320, height: 320 },
        { width: 480, height: 640 },
        { width: 640, height: 480 }
    ];
    
    return validSizes.some(size => size.width === width && size.height === height);
}

/**
 * Create detection visualization elements
 * @param {Array} detections - Detection results
 * @param {HTMLElement} container - Container element
 * @param {Object} scale - Scale factors for coordinate conversion
 */
export function visualizeDetections(detections, container, scale) {
    // Clear existing visualizations
    const existingBoxes = container.querySelectorAll('.detection-box, .landmark-point');
    existingBoxes.forEach(box => box.remove());
    
    detections.forEach((detection, index) => {
        const { bbox, confidence, landmarks } = detection;
        const [x1, y1, x2, y2] = bbox;
        
        // Create bounding box
        const box = document.createElement('div');
        box.className = 'detection-box';
        box.style.left = `${x1 * scale.x}px`;
        box.style.top = `${y1 * scale.y}px`;
        box.style.width = `${(x2 - x1) * scale.x}px`;
        box.style.height = `${(y2 - y1) * scale.y}px`;
        
        // Create label
        const label = document.createElement('div');
        label.className = 'detection-label';
        label.textContent = `Face ${Math.round(confidence * 100)}%`;
        box.appendChild(label);
        
        container.appendChild(box);
        
        // Add landmarks if available
        if (landmarks && landmarks.length >= 10) {
            const landmarkPoints = [
                { x: landmarks[0], y: landmarks[1] }, // Right eye
                { x: landmarks[2], y: landmarks[3] }, // Left eye
                { x: landmarks[4], y: landmarks[5] }, // Nose
                { x: landmarks[6], y: landmarks[7] }, // Right mouth corner
                { x: landmarks[8], y: landmarks[9] }  // Left mouth corner
            ];
            
            landmarkPoints.forEach((point, i) => {
                const landmark = document.createElement('div');
                landmark.className = 'landmark-point';
                landmark.style.left = `${point.x * scale.x}px`;
                landmark.style.top = `${point.y * scale.y}px`;
                container.appendChild(landmark);
            });
        }
    });
}
