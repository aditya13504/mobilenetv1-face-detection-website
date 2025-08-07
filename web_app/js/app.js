/**
 * Main Application Module
 * Handles camera, UI interactions, and coordinates face detection
 */

import faceDetector from './face-detector.js';
import { 
    checkBrowserSupport, 
    getCameraConstraints, 
    handleError, 
    showLoading, 
    hideLoading,
    updateStats,
    debounce,
    PerformanceMonitor
} from './utils.js';

class FaceDetectionApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isDetecting = false;
        this.animationId = null;
        this.performanceMonitor = new PerformanceMonitor();
        
        // UI elements
        this.elements = {};
        
        // Application state
        this.state = {
            cameraActive: false,
            detectionActive: false,
            currentResolution: '640x480',
            showLandmarks: true,
            showConfidence: true,
            detectionCount: 0
        };
        
        // Configuration
        this.config = {
            confidenceThreshold: 0.7,
            nmsThreshold: 0.4,
            maxDetections: 50,
            detectionInterval: 100 // ms between detections
        };
        
        this.lastDetectionTime = 0;
    }
    
    /**
     * Initialize the application
     */
    async initialize() {
        try {
            showLoading('Starting Face Detection App...');
            
            // Check browser support
            const support = checkBrowserSupport();
            this.displayBrowserSupport(support);
            
            if (!support.getUserMedia) {
                throw new Error('Camera access is not supported in this browser.');
            }
            
            // Initialize UI elements
            this.initializeElements();
            this.setupEventListeners();
            
            // Initialize face detector
            showLoading('Loading face detection model...');
            await faceDetector.initialize('./models/optimized_model.onnx', {
                useWebGL: support.webgl2 || support.webgl,
                numThreads: navigator.hardwareConcurrency || 4
            });
            
            hideLoading();
            this.updateModelStatus('Ready');
            
            console.log('Face Detection App initialized successfully');
            
        } catch (error) {
            hideLoading();
            handleError(error, 'App Initialization');
        }
    }
    
    /**
     * Initialize UI elements
     */
    initializeElements() {
        this.elements = {
            // Video and canvas
            video: document.getElementById('video'),
            canvas: document.getElementById('canvas'),
            videoWrapper: document.querySelector('.video-wrapper'),
            
            // Controls
            startBtn: document.getElementById('start-camera'),
            stopBtn: document.getElementById('stop-camera'),
            detectBtn: document.getElementById('start-detection'),
            captureBtn: document.getElementById('capture-frame'),
            resolutionSelect: document.getElementById('resolution'),
            
            // Sliders
            confidenceSlider: document.getElementById('confidence-threshold'),
            confidenceValue: document.getElementById('confidence-value'),
            nmsSlider: document.getElementById('nms-threshold'),
            nmsValue: document.getElementById('nms-value'),
            
            // Toggles
            landmarksToggle: document.getElementById('show-landmarks'),
            confidenceToggle: document.getElementById('show-confidence'),
            
            // Stats
            fps: document.getElementById('fps'),
            inferenceTime: document.getElementById('inference-time'),
            facesDetected: document.getElementById('faces-detected'),
            modelStatus: document.getElementById('model-status'),
            
            // Modals
            errorModal: document.getElementById('error-modal'),
            closeModalBtn: document.getElementById('close-modal')
        };
        
        this.video = this.elements.video;
        this.canvas = this.elements.canvas;
        this.ctx = this.canvas.getContext('2d');
        
        // Set initial values
        this.elements.confidenceValue.textContent = this.config.confidenceThreshold.toFixed(2);
        this.elements.nmsValue.textContent = this.config.nmsThreshold.toFixed(2);
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Camera controls
        this.elements.startBtn.addEventListener('click', () => this.startCamera());
        this.elements.stopBtn.addEventListener('click', () => this.stopCamera());
        this.elements.detectBtn.addEventListener('click', () => this.toggleDetection());
        this.elements.captureBtn.addEventListener('click', () => this.captureFrame());
        
        // Resolution change
        this.elements.resolutionSelect.addEventListener('change', (e) => {
            this.changeResolution(e.target.value);
        });
        
        // Sliders
        this.elements.confidenceSlider.addEventListener('input', debounce((e) => {
            this.updateConfidenceThreshold(parseFloat(e.target.value));
        }, 100));
        
        this.elements.nmsSlider.addEventListener('input', debounce((e) => {
            this.updateNMSThreshold(parseFloat(e.target.value));
        }, 100));
        
        // Toggles
        this.elements.landmarksToggle.addEventListener('change', (e) => {
            this.state.showLandmarks = e.target.checked;
        });
        
        this.elements.confidenceToggle.addEventListener('change', (e) => {
            this.state.showConfidence = e.target.checked;
        });
        
        // Modal close
        this.elements.closeModalBtn.addEventListener('click', () => {
            this.elements.errorModal.classList.remove('show');
        });
        
        // Video events
        this.video.addEventListener('loadedmetadata', () => this.onVideoLoaded());
        this.video.addEventListener('error', (e) => handleError(e, 'Video'));
        
        // Window events
        window.addEventListener('beforeunload', () => this.cleanup());
        window.addEventListener('resize', debounce(() => this.updateCanvasSize(), 250));
    }
    
    /**
     * Start camera stream
     */
    async startCamera() {
        try {
            showLoading('Starting camera...');
            
            const constraints = getCameraConstraints(this.state.currentResolution);
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            this.video.srcObject = this.stream;
            await this.video.play();
            
            this.state.cameraActive = true;
            this.updateCameraControls();
            
            hideLoading();
            console.log('Camera started successfully');
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Camera Start');
        }
    }
    
    /**
     * Stop camera stream
     */
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        this.state.cameraActive = false;
        
        this.stopDetection();
        this.updateCameraControls();
        
        console.log('Camera stopped');
    }
    
    /**
     * Toggle face detection
     */
    toggleDetection() {
        if (this.isDetecting) {
            this.stopDetection();
        } else {
            this.startDetection();
        }
    }
    
    /**
     * Start face detection
     */
    startDetection() {
        if (!this.state.cameraActive || this.isDetecting) {
            return;
        }
        
        this.isDetecting = true;
        this.state.detectionActive = true;
        this.updateDetectionControls();
        
        // Start detection loop
        this.detectLoop();
        
        console.log('Face detection started');
    }
    
    /**
     * Stop face detection
     */
    stopDetection() {
        this.isDetecting = false;
        this.state.detectionActive = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        // Clear canvas
        this.clearCanvas();
        this.updateDetectionControls();
        
        console.log('Face detection stopped');
    }
    
    /**
     * Main detection loop
     */
    detectLoop() {
        if (!this.isDetecting) return;
        
        const now = performance.now();
        
        // Throttle detections based on interval
        if (now - this.lastDetectionTime >= this.config.detectionInterval) {
            this.performDetection();
            this.lastDetectionTime = now;
        }
        
        // Update FPS
        this.performanceMonitor.recordFrame();
        
        this.animationId = requestAnimationFrame(() => this.detectLoop());
    }
    
    /**
     * Perform face detection on current video frame
     */
    async performDetection() {
        if (!this.video.videoWidth || !this.video.videoHeight) {
            return;
        }
        
        try {
            // Capture current frame
            const imageData = this.captureVideoFrame();
            
            // Run face detection
            const result = await faceDetector.detect(imageData);
            const { detections, inferenceTime, stats } = result;
            
            // Update detection count
            this.state.detectionCount = detections.length;
            
            // Draw results
            this.drawDetections(detections);
            
            // Update stats
            this.updateStatsDisplay(stats, detections.length);
            
        } catch (error) {
            console.error('Detection error:', error);
            // Don't stop detection for individual frame errors
        }
    }
    
    /**
     * Capture current video frame as ImageData
     */
    captureVideoFrame() {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        
        tempCtx.drawImage(this.video, 0, 0);
        
        return tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    }
    
    /**
     * Draw detection results on canvas
     */
    drawDetections(detections) {
        // Clear previous drawings
        this.clearCanvas();
        
        if (!detections || detections.length === 0) {
            return;
        }
        
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.fillStyle = '#10b981';
        this.ctx.lineWidth = 3;
        this.ctx.font = '14px Arial';
        
        detections.forEach((detection, index) => {
            const { bbox, confidence, landmarks } = detection;
            const [x1, y1, x2, y2] = bbox;
            
            // Scale coordinates
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;
            
            // Draw bounding box
            this.ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
            
            // Draw confidence label
            if (this.state.showConfidence) {
                const label = `Face ${Math.round(confidence * 100)}%`;
                const labelWidth = this.ctx.measureText(label).width;
                
                // Background
                this.ctx.fillStyle = '#10b981';
                this.ctx.fillRect(sx1, sy1 - 25, labelWidth + 10, 20);
                
                // Text
                this.ctx.fillStyle = 'white';
                this.ctx.fillText(label, sx1 + 5, sy1 - 10);
            }
            
            // Draw landmarks
            if (this.state.showLandmarks && landmarks && landmarks.length >= 10) {
                this.ctx.fillStyle = '#f59e0b';
                for (let i = 0; i < landmarks.length; i += 2) {
                    const lx = landmarks[i] * scaleX;
                    const ly = landmarks[i + 1] * scaleY;
                    
                    this.ctx.beginPath();
                    this.ctx.arc(lx, ly, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                }
            }
        });
    }
    
    /**
     * Clear canvas
     */
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    /**
     * Capture and download current frame
     */
    captureFrame() {
        if (!this.state.cameraActive) {
            return;
        }
        
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        captureCanvas.width = this.video.videoWidth;
        captureCanvas.height = this.video.videoHeight;
        
        // Draw video frame
        captureCtx.drawImage(this.video, 0, 0);
        
        // Draw detections if active
        if (this.state.detectionActive) {
            const scaleX = captureCanvas.width / this.canvas.width;
            const scaleY = captureCanvas.height / this.canvas.height;
            
            captureCtx.scale(scaleX, scaleY);
            captureCtx.drawImage(this.canvas, 0, 0);
        }
        
        // Download image
        captureCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `face-detection-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
    
    /**
     * Change camera resolution
     */
    async changeResolution(resolution) {
        if (!this.state.cameraActive) {
            this.state.currentResolution = resolution;
            return;
        }
        
        try {
            showLoading('Changing resolution...');
            
            // Stop current stream
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            
            // Start with new resolution
            const constraints = getCameraConstraints(resolution);
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.state.currentResolution = resolution;
            hideLoading();
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Resolution Change');
        }
    }
    
    /**
     * Update confidence threshold
     */
    updateConfidenceThreshold(value) {
        this.config.confidenceThreshold = value;
        this.elements.confidenceValue.textContent = value.toFixed(2);
        
        faceDetector.updateConfig({
            confidenceThreshold: value
        });
    }
    
    /**
     * Update NMS threshold
     */
    updateNMSThreshold(value) {
        this.config.nmsThreshold = value;
        this.elements.nmsValue.textContent = value.toFixed(2);
        
        faceDetector.updateConfig({
            nmsThreshold: value
        });
    }
    
    /**
     * Video loaded event handler
     */
    onVideoLoaded() {
        this.updateCanvasSize();
        console.log(`Video loaded: ${this.video.videoWidth}x${this.video.videoHeight}`);
    }
    
    /**
     * Update canvas size to match video
     */
    updateCanvasSize() {
        if (this.video.videoWidth && this.video.videoHeight) {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
        }
    }
    
    /**
     * Update camera control buttons
     */
    updateCameraControls() {
        this.elements.startBtn.disabled = this.state.cameraActive;
        this.elements.stopBtn.disabled = !this.state.cameraActive;
        this.elements.detectBtn.disabled = !this.state.cameraActive;
        this.elements.captureBtn.disabled = !this.state.cameraActive;
    }
    
    /**
     * Update detection control buttons
     */
    updateDetectionControls() {
        this.elements.detectBtn.textContent = this.state.detectionActive ? 'Stop Detection' : 'Start Detection';
        this.elements.detectBtn.className = this.state.detectionActive ? 'btn btn-secondary' : 'btn btn-primary';
    }
    
    /**
     * Update statistics display
     */
    updateStatsDisplay(stats, faceCount) {
        updateStats({
            fps: this.performanceMonitor.getFPS(),
            inferenceTime: `${stats.avgInferenceTime}ms`,
            facesDetected: faceCount,
            modelStatus: 'Running'
        });
    }
    
    /**
     * Update model status
     */
    updateModelStatus(status) {
        if (this.elements.modelStatus) {
            this.elements.modelStatus.textContent = status;
        }
    }
    
    /**
     * Display browser support information
     */
    displayBrowserSupport(support) {
        console.log('Browser Support:', support);
        
        if (!support.webgl && !support.webgl2) {
            console.warn('WebGL not supported - using CPU inference');
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopCamera();
        faceDetector.dispose();
        console.log('App cleanup completed');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new FaceDetectionApp();
    await app.initialize();
    
    // Make app available globally for debugging
    window.faceDetectionApp = app;
});
