"""
Quick Test Script - Non-Interactive.
Langsung test dengan dummy image tanpa perlu input manual.

Usage: python quick_test.py
"""

import sys
import os
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PotholeDetector
from config import Config
from test_detector import create_dummy_pothole_image


def main():
    """Quick test dengan dummy image."""
    print("="*70)
    print("🚗 QUICK TEST - POTHOLE DETECTION")
    print("="*70)
    
    # Initialize
    print("\n⚙️ Initializing detector...")
    config = Config()
    config.DEBUG_MODE = False
    detector = PotholeDetector(config)
    
    # Create dummy image
    print("\n🎨 Creating test image...")
    image = create_dummy_pothole_image()
    
    # Save original
    os.makedirs('../output/images', exist_ok=True)
    cv2.imwrite('../output/images/quick_test_original.jpg', image)
    
    # Detect
    print("\n🔍 Running detection...")
    results = detector.detect_single_image(image)
    
    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"✅ Potholes detected: {results['num_potholes']}")
    print(f"⏱️ Processing time: {results['processing_time']:.3f}s")
    
    if results['num_potholes'] > 0:
        print(f"\n📦 Details:")
        for i, (x, y, w, h) in enumerate(results['potholes']):
            print(f"   Pothole #{i+1}: ({x}, {y}) - {w}×{h} pixels")
    
    # Save result
    detector.save_results(image, results, '../output/images/quick_test_result.jpg')
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED!")
    print("="*70)
    print(f"📂 Check results:")
    print(f"   - output/images/quick_test_original.jpg")
    print(f"   - output/images/quick_test_result.jpg")
    print("="*70)


if __name__ == "__main__":
    main()