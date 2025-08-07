#!/usr/bin/env python3
"""
RetinaFace ONNX Inference Example
This script demonstrates how to run inference using the optimized ONNX model
"""

import numpy as np
import onnxruntime as ort
import cv2
import os

class RetinaFaceONNX:
    def __init__(self, model_path):
        """
        Initialize RetinaFace ONNX model
        
        Args:
            model_path (str): Path to ONNX model file
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input/output details
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        
        print(f"Model loaded: {model_path}")
        print(f"Input: {self.input_details[0].name} {self.input_details[0].shape}")
        print(f"Outputs: {len(self.output_details)}")
    
    def preprocess_image(self, image_path, target_size=(640, 640)):
        """
        Load and preprocess image for RetinaFace model
        
        Args:
            image_path (str): Path to input image
            target_size (tuple): Target size for resizing
        
        Returns:
            tuple: (preprocessed_image, original_image, scale_factor)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_img = img.copy()
        
        # Calculate scale factor
        h, w = img.shape[:2]
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Convert to float32 and normalize
        img = img.astype(np.float32)
        
        # Subtract mean (BGR order: 104, 117, 123)
        img -= np.array([104, 117, 123])
        
        # Convert from HWC to CHW (ONNX format)
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, original_img, (scale_x, scale_y)
    
    def run_inference(self, preprocessed_image):
        """
        Run inference on preprocessed image
        
        Args:
            preprocessed_image (np.ndarray): Preprocessed image array
        
        Returns:
            list: Model outputs [bbox_regressions, classifications, landmark_regressions]
        """
        input_name = self.input_details[0].name
        outputs = self.session.run(None, {input_name: preprocessed_image})
        return outputs
    
    def detect_faces(self, image_path, confidence_threshold=0.5):
        """
        Complete face detection pipeline
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Confidence threshold for detections
        
        Returns:
            dict: Detection results with bounding boxes, landmarks, and scores
        """
        # Preprocess image
        preprocessed_img, original_img, scale = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.run_inference(preprocessed_img)
        
        bbox_regressions = outputs[0]  # Shape: (1, N, 4)
        classifications = outputs[1]   # Shape: (1, N, 2)  
        landmark_regressions = outputs[2]  # Shape: (1, N, 10)
        
        print(f"Raw output shapes:")
        print(f"  Bbox regressions: {bbox_regressions.shape}")
        print(f"  Classifications: {classifications.shape}")
        print(f"  Landmark regressions: {landmark_regressions.shape}")
        
        # Basic post-processing (simplified)
        # Note: Complete implementation requires prior box generation and NMS
        results = {
            'bbox_regressions': bbox_regressions,
            'classifications': classifications,
            'landmark_regressions': landmark_regressions,
            'original_image': original_img,
            'scale': scale
        }
        
        return results

def benchmark_model(model_path, num_runs=10):
    """
    Benchmark model inference speed
    
    Args:
        model_path (str): Path to ONNX model
        num_runs (int): Number of inference runs
    """
    detector = RetinaFaceONNX(model_path)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        _ = detector.run_inference(dummy_input)
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = detector.run_inference(dummy_input)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"\nBenchmark Results:")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {1/avg_time:.2f}")

def visualize_detections(results, save_path=None):
    """
    Visualize detection results (basic implementation)
    
    Args:
        results (dict): Detection results from detect_faces
        save_path (str): Path to save visualization
    """
    original_img = results['original_image']
    
    # This is a placeholder - actual visualization requires:
    # 1. Proper decoding of bounding boxes
    # 2. Confidence filtering  
    # 3. Non-maximum suppression
    # 4. Coordinate transformation
    
    print("\nVisualization:")
    print("- Original image shape:", original_img.shape)
    print("- To implement proper visualization, you need to:")
    print("  1. Generate prior/anchor boxes")
    print("  2. Decode bounding box predictions")
    print("  3. Apply confidence threshold")
    print("  4. Perform non-maximum suppression")
    print("  5. Transform coordinates back to original image size")
    
    if save_path:
        cv2.imwrite(save_path, original_img)
        print(f"- Original image saved to: {save_path}")

def main():
    # Configuration
    model_path = "optimized_model.onnx"  # or "model.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Available models:")
        for file in os.listdir("."):
            if file.endswith(".onnx"):
                print(f"  - {file}")
        return
    
    # Initialize detector
    print("Initializing RetinaFace detector...")
    detector = RetinaFaceONNX(model_path)
    
    # Benchmark model
    print("\nRunning benchmark...")
    benchmark_model(model_path)
    
    # Example with dummy image (since we don't have a real image)
    print("\nExample inference with dummy data:")
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = detector.run_inference(dummy_input)
    
    print("Inference completed successfully!")
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Output {i} shape: {output.shape}")
    
    # If you have a real image, uncomment and modify this:
    # image_path = "path/to/your/image.jpg"
    # if os.path.exists(image_path):
    #     print(f"\nProcessing image: {image_path}")
    #     results = detector.detect_faces(image_path)
    #     visualize_detections(results, "detection_result.jpg")
    
    print("\n" + "="*50)
    print("IMPORTANT NOTES:")
    print("="*50)
    print("This example shows basic ONNX inference.")
    print("For complete face detection, you need to implement:")
    print("1. Prior/anchor box generation")
    print("2. Bounding box regression decoding")
    print("3. Non-maximum suppression (NMS)")
    print("4. Coordinate transformation")
    print("5. Confidence filtering")
    print("\nSee RetinaFace paper and original implementation for details:")
    print("https://arxiv.org/abs/1905.00641")

if __name__ == "__main__":
    main()
