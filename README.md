# Real-Time Face Detection Web Application

A high-performance, browser-based web application for real-time face detection using an optimized MobileNet 0.25 ONNX model and ONNX Runtime (ONNX.js) with WebGL acceleration.

---

## ✨ Features

- **Real-time face detection** from your webcam
- **High performance**: 46+ FPS with optimized ONNX model
- **Runs entirely in the browser** (no data leaves your device)
- **ONNX.js** for GPU-accelerated inference (WebGL)
- **Configurable detection parameters**: confidence threshold, NMS threshold
- **Visual feedback**: bounding boxes, confidence scores, facial landmarks
- **Responsive UI**: works on desktop and mobile
- **Frame capture**: save detected frames as images
- **Performance monitoring**: FPS, inference time, faces detected

---

## 🏗️ Project Structure

```
FaceDetector.onnx                # (Legacy/Reference) ONNX model
serve_web_app.py                 # Python HTTP server for local development
onnx_optimized/
    DEPLOYMENT_GUIDE.md          # Model deployment guide
    optimized_model.onnx         # Optimized ONNX model (RetinaFace)
    retinaface_inference.py      # Python inference script
web_app/
    index.html                   # Main application UI
    test.html                    # Model loading/inference test page
    README.md                    # (This file)
    css/
        style.css                # App styling
    js/
        app.js                   # Main app logic (UI, camera, rendering)
        face-detector.js         # ONNX model integration, inference
        simple-app.js            # (Optional) Minimal demo logic
        utils.js                 # Utility functions
    models/
        optimized_model.onnx     # Face detection model (1.65MB)
```

---

## 🚀 Quick Start

### Prerequisites
- Modern web browser (Chrome 80+, Firefox 75+, Safari 13.1+, Edge 80+)
- Camera/webcam connected to your device
- Python 3.6+ (for local server)

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aditya13504/mobilenetv1-face-detection-website.git
cd mobilenetv1-face-detection-website/web_app
```

---

## 🖥️ Local Development

You can run the app locally using Python:

```bash
python ../serve_web_app.py
```

Or use any static server (e.g., `npx http-server . -p 8000 --cors`)

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🌐 Vercel Deployment

This project is ready for static deployment on Vercel.

1. **Ensure the following structure:**
   - All web files are inside the `web_app/` directory.
   - The ONNX model is in `web_app/models/optimized_model.onnx`.
2. **Add `vercel.json`** (already included):
   - Serves all files from `web_app/` as static assets.
3. **Deploy:**
   - Push to GitHub and connect the repo to Vercel.
   - Or use `vercel` CLI:
     ```bash
     vercel --prod
     ```

---

## 🎛️ Controls & Settings

### Camera Controls
- **Start/Stop Camera**: Control video stream
- **Resolution**: Select camera resolution (360p to 1080p)
- **Capture Frame**: Save current frame with detections

### Detection Settings
- **Confidence Threshold** (0.1-0.9): Minimum confidence for face detection
- **NMS Threshold** (0.1-0.9): Non-maximum suppression for overlapping detections
- **Show Landmarks**: Toggle facial landmark visualization
- **Show Confidence**: Toggle confidence score display

### Performance Monitoring
- **FPS**: Real-time frames per second
- **Inference Time**: Model processing time per frame
- **Faces Detected**: Number of faces in current frame
- **Model Status**: Current model state

---

## 🔧 Technical Details

### Model
- **Architecture**: RetinaFace (MobileNet0.25 backbone)
- **Input Size**: 640×640 pixels
- **Model Size**: 1.65MB (optimized)
- **Outputs**: Bounding boxes, confidence scores, facial landmarks

### Inference
- **ONNX Runtime (onnxruntime-web)**: WebGL backend for GPU acceleration
- **Efficient rendering**: Canvas 2D API
- **Configurable detection intervals**: Balance performance and accuracy

### Browser Requirements
- **WebGL Support** (for GPU acceleration)
- **Camera API**: `navigator.mediaDevices.getUserMedia`
- **Modern browser** (see above)

---

## 🎯 Usage Examples

### Basic Face Detection
1. Start camera and detection
2. Default settings work well for most scenarios
3. Faces are highlighted with green bounding boxes

### High-Accuracy Detection
- Increase confidence threshold to 0.8+
- Reduce NMS threshold to 0.3
- Enable landmark visualization

### Performance Optimization
- Use lower camera resolution (480p or 360p)
- Increase detection interval in code if needed
- Disable landmarks for faster rendering

---

## 📊 Performance Benchmarks

| Device Type         | FPS  | Inference Time | Notes                  |
|--------------------|------|---------------|------------------------|
| Desktop (GPU)      | 60+  | 15-20ms       | WebGL acceleration     |
| Desktop (CPU)      | 30-45| 25-35ms       | WASM fallback          |
| Mobile (High-end)  | 25-35| 30-40ms       | Variable performance   |
| Mobile (Mid-range) | 15-25| 40-60ms       | Reduced resolution rec |

---

## 🔐 Privacy & Security

- **Local Processing**: All detection runs in your browser
- **No Data Upload**: Camera feed never leaves your device
- **No Storage**: Frames are processed in memory only
- **HTTPS Recommended**: For camera access on remote servers

---

## 🛠️ Development

- Main logic: `web_app/js/app.js`, `web_app/js/face-detector.js`
- UI: `web_app/index.html`, `web_app/css/style.css`
- Model: `web_app/models/optimized_model.onnx`
- Utilities: `web_app/js/utils.js`
- Test page: `web_app/test.html`

### Model Replacement
To use a different ONNX model:
1. Replace `models/optimized_model.onnx`
2. Update input/output handling in `face-detector.js`
3. Adjust preprocessing if needed

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **RetinaFace** - Original face detection architecture
- **ONNX Runtime** - Cross-platform inference engine
- **MobileNet** - Efficient neural network architecture
- **ONNX.js** - JavaScript runtime for ONNX models

---

**🎉 Enjoy real-time face detection in your browser!**
