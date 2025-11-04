"""
Main Detector Module - PotholeDetector Class.
Mengintegrasikan semua tahapan pipeline deteksi lubang jalan:
1. Preprocessing (Resize + Grayscale)
2. Segmentation (Adaptive Threshold)
3. Edge Detection (Canny)
4. Post-Processing (Dilasi + Filtering)

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

import cv2
import numpy as np
import time
from config import Config
from preprocessor import Preprocessor
from segmentation import Segmentation
from edge_detection import EdgeDetection
from postprocessing import PostProcessing


class PotholeDetector:
    """
    Main class untuk deteksi lubang jalan.
    Mengintegrasikan seluruh pipeline image processing.
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi PotholeDetector dengan semua module.
        
        Parameters:
        -----------
        config : Config object, optional
            Objek konfigurasi. Jika None, gunakan Config default.
        
        Example:
        --------
        >>> detector = PotholeDetector()
        >>> detector = PotholeDetector(custom_config)
        """
        self.config = config if config else Config()
        
        # Initialize all modules
        self.preprocessor = Preprocessor(self.config)
        self.segmenter = Segmentation(self.config)
        self.edge_detector = EdgeDetection(self.config)
        self.postprocessor = PostProcessing(self.config)
        
        if self.config.DEBUG_MODE:
            print("✅ PotholeDetector initialized successfully!")
            print(f"   - Preprocessor ready")
            print(f"   - Segmenter ready")
            print(f"   - Edge Detector ready")
            print(f"   - Post-Processor ready")
    
    
    def detect_single_image(self, image):
        """
        Deteksi lubang pada single image.
        
        Pipeline Lengkap (5 Tahapan):
        1. Resize → 640×480
        2. Grayscale → 1 channel
        3. Adaptive Threshold → Binary
        4. Canny Edge Detection → Edge map
        5. Post-Processing → Filtered contours
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (RGB/BGR)
            Shape: (height, width, 3)
        
        Returns:
        --------
        results : dict
            Dictionary berisi semua hasil:
            {
                'potholes': list of bounding boxes [(x,y,w,h), ...],
                'num_potholes': int,
                'processing_time': float (seconds),
                'intermediate_images': dict (jika SAVE_INTERMEDIATE=True)
            }
        
        Example:
        --------
        >>> image = cv2.imread('jalan_berlubang.jpg')
        >>> detector = PotholeDetector()
        >>> results = detector.detect_single_image(image)
        >>> print(f"Found {results['num_potholes']} potholes")
        >>> print(f"Processing time: {results['processing_time']:.2f}s")
        """
        start_time = time.time()
        
        intermediate_images = {}
        
        try:
            # ===== TAHAP 1 & 2: PREPROCESSING =====
            if self.config.DEBUG_MODE:
                print("\n" + "="*60)
                print("🎯 STARTING POTHOLE DETECTION PIPELINE")
                print("="*60)
            
            gray = self.preprocessor.preprocess(image)
            if self.config.SAVE_INTERMEDIATE:
                intermediate_images['grayscale'] = gray
            
            # ===== TAHAP 3: SEGMENTATION =====
            binary = self.segmenter.adaptive_threshold(gray)
            if self.config.SAVE_INTERMEDIATE:
                intermediate_images['binary'] = binary
            
            # ===== TAHAP 4: EDGE DETECTION =====
            edges = self.edge_detector.canny_edge_detection(binary)
            if self.config.SAVE_INTERMEDIATE:
                intermediate_images['edges'] = edges
            
            # ===== TAHAP 5: POST-PROCESSING =====
            contours, bounding_boxes, dilated = self.postprocessor.process(edges)
            if self.config.SAVE_INTERMEDIATE:
                intermediate_images['dilated'] = dilated
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare results
            results = {
                'potholes': bounding_boxes,
                'num_potholes': len(bounding_boxes),
                'processing_time': processing_time,
                'intermediate_images': intermediate_images,
                'contours': contours  # For advanced visualization
            }
            
            if self.config.DEBUG_MODE:
                print(f"\n🎉 DETECTION COMPLETED!")
                print(f"   Potholes detected: {results['num_potholes']}")
                print(f"   Processing time: {processing_time:.3f}s")
                print("="*60 + "\n")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'potholes': [],
                'num_potholes': 0,
                'processing_time': time.time() - start_time,
                'intermediate_images': intermediate_images,
                'error': str(e)
            }
    
    
    def draw_results(self, image, results):
        """
        Gambar bounding boxes pada image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Original image
        results : dict
            Results dari detect_single_image()
        
        Returns:
        --------
        annotated_image : numpy.ndarray
            Image dengan bounding boxes dan labels
        
        Example:
        --------
        >>> results = detector.detect_single_image(image)
        >>> annotated = detector.draw_results(image, results)
        >>> cv2.imshow('Detection', annotated)
        """
        # Make copy to avoid modifying original
        annotated = image.copy()
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(results['potholes']):
            # Draw rectangle
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                self.config.BBOX_COLOR,
                self.config.BBOX_THICKNESS
            )
            
            # Draw label
            label = f"Pothole #{i+1}"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                self.config.FONT_THICKNESS
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                self.config.BBOX_COLOR,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                (255, 255, 255),  # White text
                self.config.FONT_THICKNESS
            )
        
        # Draw summary info
        summary = f"Detected: {results['num_potholes']} potholes | Time: {results['processing_time']:.2f}s"
        cv2.putText(
            annotated,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2
        )
        
        return annotated
    
    
    def save_results(self, image, results, output_path):
        """
        Simpan hasil deteksi ke file.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Original image
        results : dict
            Results dari detect_single_image()
        output_path : str
            Path output file
        
        Returns:
        --------
        success : bool
            True jika berhasil save
        
        Example:
        --------
        >>> results = detector.detect_single_image(image)
        >>> detector.save_results(image, results, 'output/result.jpg')
        """
        annotated = self.draw_results(image, results)
        success = cv2.imwrite(output_path, annotated)
        
        if success:
            print(f"✅ Results saved: {output_path}")
        else:
            print(f"❌ Failed to save: {output_path}")
        
        return success
    
    
    def print_results(self, results):
        """
        Print hasil deteksi dalam format yang readable.
        
        Parameters:
        -----------
        results : dict
            Results dari detect_single_image()
        
        Example:
        --------
        >>> results = detector.detect_single_image(image)
        >>> detector.print_results(results)
        """
        print("\n" + "="*60)
        print("📊 DETECTION RESULTS")
        print("="*60)
        print(f"\n🎯 Summary:")
        print(f"   Total potholes detected: {results['num_potholes']}")
        print(f"   Processing time: {results['processing_time']:.3f} seconds")
        
        if results['num_potholes'] > 0:
            print(f"\n📦 Pothole Details:")
            for i, (x, y, w, h) in enumerate(results['potholes']):
                area = w * h
                print(f"\n   Pothole #{i+1}:")
                print(f"      Location: (x={x}, y={y})")
                print(f"      Size: {w} × {h} pixels")
                print(f"      Area: {area} pixels²")
        else:
            print(f"\n   ℹ️ No potholes detected in this image")
        
        print("\n" + "="*60)


# ========== HELPER FUNCTIONS ==========

def load_image(image_path):
    """Load image dari file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
    return image


# ========== TEST CODE ==========

if __name__ == "__main__":
    """
    Test code untuk memverifikasi PotholeDetector bekerja dengan benar.
    """
    print("🧪 TESTING POTHOLE DETECTOR")
    print("="*60)
    
    # Create dummy test image
    print("\n1️⃣ Creating test image...")
    test_image = np.random.randint(150, 200, (800, 1200, 3), dtype=np.uint8)
    
    # Add simulated potholes (dark spots)
    # Pothole 1: Large circular hole
    cv2.circle(test_image, (300, 300), 40, (50, 50, 50), -1)
    
    # Pothole 2: Smaller oval hole
    cv2.ellipse(test_image, (700, 400), (30, 20), 0, 0, 360, (60, 60, 60), -1)
    
    # Add noise (small dark spots that should be filtered out)
    cv2.circle(test_image, (500, 200), 5, (40, 40, 40), -1)
    
    print(f"   Test image created: {test_image.shape}")
    print(f"   Simulated 2 potholes + 1 noise")
    
    # Initialize detector
    print("\n2️⃣ Initializing PotholeDetector...")
    config = Config()
    config.DEBUG_MODE = True
    detector = PotholeDetector(config)
    
    # Run detection
    print("\n3️⃣ Running detection...")
    results = detector.detect_single_image(test_image)
    
    # Print results
    detector.print_results(results)
    
    # Test visualization
    print("\n4️⃣ Testing visualization...")
    annotated = detector.draw_results(test_image, results)
    print(f"   Annotated image shape: {annotated.shape}")
    
    # Final summary
    print("\n" + "="*60)
    if results['num_potholes'] >= 2:
        print("✅ TEST PASSED! Detected potholes correctly")
    else:
        print(f"⚠️ TEST WARNING: Expected 2 potholes, detected {results['num_potholes']}")
    print("="*60)