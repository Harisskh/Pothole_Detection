"""
Simple Test Script untuk PotholeDetector.
Script ini bisa test dengan:
1. Foto dummy (simulasi)
2. Foto real dari dataset

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PotholeDetector
from config import Config

def test_with_real_image(detector, image_path, save_output=True):
    """
    Test detector dengan real image dari dataset.
    
    Parameters:
    -----------
    detector : PotholeDetector
        Instance detector
    image_path : str
        Path ke foto real
    save_output : bool
        Simpan hasil atau tidak
    
    Returns:
    --------
    results : dict
        Detection results
    """
    print("\n" + "="*70)
    print("🧪 TEST WITH REAL IMAGE")
    print("="*70)
    
    # Load image
    print(f"\n📂 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"   ❌ Failed to load image!")
        return None
    
    print(f"   ✅ Image loaded: {image.shape}")
    
    # Run detection
    print(f"\n🔍 Running detection...")
    results = detector.detect_single_image(image)
    
    # Print results
    detector.print_results(results)
    
    # Save annotated result
    if save_output:
        os.makedirs('../output/images', exist_ok=True)
        output_name = os.path.basename(image_path).replace('.', '_result.')
        output_path = f'../output/images/{output_name}'
        detector.save_results(image, results, output_path)
    
    return results


def test_batch_images(detector, folder_path):
    """
    Test detector dengan batch images dari folder.
    
    Parameters:
    -----------
    detector : PotholeDetector
        Instance detector
    folder_path : str
        Path ke folder berisi images
    
    Returns:
    --------
    all_results : list
        List of all detection results
    """
    print("\n" + "="*70)
    print("🧪 BATCH TEST WITH FOLDER")
    print("="*70)
    
    # Get all image files
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                image_files.append(os.path.join(folder_path, file))
    else:
        print(f"   ❌ Folder not found: {folder_path}")
        return []
    
    print(f"\n📂 Found {len(image_files)} images in folder")
    
    if len(image_files) == 0:
        print("   ⚠️ No images to test!")
        return []
    
    # Process each image
    all_results = []
    total_potholes = 0
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        print(f"\n{'─'*70}")
        print(f"Processing [{i+1}/{len(image_files)}]: {os.path.basename(img_path)}")
        
        results = test_with_real_image(detector, img_path, save_output=True)
        
        if results:
            all_results.append(results)
            total_potholes += results['num_potholes']
            total_time += results['processing_time']
    
    # Summary
    print("\n" + "="*70)
    print("📊 BATCH TEST SUMMARY")
    print("="*70)
    print(f"   Total images processed: {len(all_results)}")
    print(f"   Total potholes detected: {total_potholes}")
    print(f"   Total processing time: {total_time:.2f}s")
    print(f"   Average time per image: {total_time/len(all_results):.3f}s")
    print(f"   Average potholes per image: {total_potholes/len(all_results):.1f}")
    print("="*70)
    
    return all_results


def main():
    """
    Main function untuk testing.
    """
    print("="*70)
    print("🚗 POTHOLE DETECTION SYSTEM - TESTING SCRIPT")
    print("="*70)
    
    # Initialize detector
    print("\n⚙️ Initializing detector...")
    config = Config()
    config.DEBUG_MODE = False  # Set False untuk output lebih bersih
    detector = PotholeDetector(config)
    print("   ✅ Detector ready!")
    
    # Menu
    print("\n" + "="*70)
    print("📋 TEST OPTIONS:")
    print("="*70)
    print("   1. Test with dummy image (simulasi)")
    print("   2. Test with single real image")
    print("   3. Test with folder (batch testing)")
    print("   0. Exit")
    print("="*70)
    
    while True:
        choice = input("\n👉 Choose option (0-3): ").strip()
        
        if choice == '0':
            print("\n👋 Exiting... Goodbye!")
            break
        
        elif choice == '1':
            test_with_dummy_image(detector)
            print("\n✅ Check results in: output/images/")
        
        elif choice == '2':
            image_path = input("📂 Enter image path: ").strip()
            if os.path.exists(image_path):
                test_with_real_image(detector, image_path)
            else:
                print(f"   ❌ File not found: {image_path}")
        
        elif choice == '3':
            folder_path = input("📂 Enter folder path: ").strip()
            test_batch_images(detector, folder_path)
            print("\n✅ Check results in: output/images/")
        
        else:
            print("   ⚠️ Invalid choice! Please choose 0-3")
        
        # Ask to continue
        cont = input("\n🔄 Continue testing? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n👋 Exiting... Goodbye!")
            break


if __name__ == "__main__":
    main()