"""
Preprocessor Module untuk Sistem Deteksi Lubang Jalan.
Berisi fungsi-fungsi untuk preprocessing citra:
1. Resize image (dengan mempertahankan aspect ratio)
2. Grayscale conversion (RGB → Grayscale)

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

import cv2
import numpy as np
from config import Config


class Preprocessor:
    """
    Kelas untuk melakukan preprocessing pada citra input.
    Tahapan:
    1. Resize image → efisiensi komputasi
    2. Grayscale conversion → simplifikasi dari 3 channel ke 1 channel
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi Preprocessor.
        
        Parameters:
        -----------
        config : Config object, optional
            Objek konfigurasi. Jika None, akan menggunakan Config default.
        
        Example:
        --------
        >>> preprocessor = Preprocessor()
        >>> preprocessor = Preprocessor(custom_config)
        """
        self.config = config if config else Config()
        
        # Mapping string ke OpenCV interpolation method
        self.interpolation_methods = {
            'INTER_NEAREST': cv2.INTER_NEAREST,  # Paling cepat
            'INTER_LINEAR': cv2.INTER_LINEAR,    # Lebih smooth
            'INTER_CUBIC': cv2.INTER_CUBIC,      # Paling smooth, paling lambat
            'INTER_AREA': cv2.INTER_AREA         # Bagus untuk downscaling
        }
    
    
    def resize_image(self, image):
        """
        Resize image ke target size dengan mempertahankan aspect ratio.
        
        Kenapa resize?
        - Foto dari kamera HP biasanya resolusi tinggi (4000×3000 atau lebih)
        - Proses image processing lambat untuk resolusi tinggi
        - 640×480 cukup untuk deteksi lubang (detail lubang masih terlihat)
        
        Algoritma:
        - Menggunakan Nearest Neighbor Interpolation (default)
        - Aspect ratio dipertahankan (tidak distorsi)
        - Ukuran akhir bisa sedikit berbeda dari target jika aspect ratio berbeda
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (bisa RGB atau BGR dari cv2.imread)
            Shape: (height, width, 3) atau (height, width)
        
        Returns:
        --------
        resized_image : numpy.ndarray
            Image yang sudah di-resize
            Shape: mendekati (480, 640, 3) tergantung aspect ratio original
        
        Example:
        --------
        >>> image = cv2.imread('jalan.jpg')  # Shape: (3000, 4000, 3)
        >>> resized = preprocessor.resize_image(image)  # Shape: (480, 640, 3)
        
        Notes:
        ------
        - Jika input sudah kecil dari target size, tidak akan di-upscale
        - Mempertahankan aspect ratio untuk menghindari distorsi
        """
        if image is None:
            raise ValueError("Input image is None. Check if image loaded correctly.")
        
        # Dapatkan dimensi original
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.config.TARGET_SIZE
        
        # Jika gambar sudah lebih kecil dari target, skip resize
        if original_width <= target_width and original_height <= target_height:
            if self.config.DEBUG_MODE:
                print(f"⚠️ Image sudah kecil ({original_width}x{original_height}), skip resize")
            return image.copy()
        
        # Hitung aspect ratio
        aspect_ratio = original_width / original_height
        
        # Tentukan ukuran baru dengan mempertahankan aspect ratio
        if aspect_ratio > 1:  # Landscape (lebih lebar)
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:  # Portrait atau square
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Pastikan tidak melebihi target
        if new_width > target_width:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        if new_height > target_height:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Get interpolation method
        interp_method = self.interpolation_methods.get(
            self.config.RESIZE_INTERPOLATION, 
            cv2.INTER_NEAREST
        )
        
        # Resize image
        resized_image = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=interp_method
        )
        
        if self.config.DEBUG_MODE:
            print(f"✅ Resize: {original_width}x{original_height} → {new_width}x{new_height}")
            print(f"   Aspect ratio preserved: {aspect_ratio:.2f}")
        
        return resized_image
    
    
    def convert_to_grayscale(self, image):
        """
        Konversi citra RGB/BGR ke Grayscale.
        
        Kenapa grayscale?
        - Simplifikasi: 3 channel (RGB) → 1 channel (Gray)
        - Efisiensi: proses lebih cepat
        - Fokus ke intensitas: lubang = area gelap, jalan = area terang
        - Warna tidak penting untuk deteksi lubang
        
        Formula:
        Gray = 0.2989*R + 0.5870*G + 0.0721*B
        
        Formula ini memberikan weight lebih besar pada hijau karena mata manusia
        lebih sensitif terhadap hijau.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (RGB atau BGR)
            Shape: (height, width, 3)
        
        Returns:
        --------
        gray_image : numpy.ndarray
            Grayscale image
            Shape: (height, width)
            Dtype: uint8 (range 0-255)
        
        Example:
        --------
        >>> rgb_image = cv2.imread('jalan.jpg')  # Shape: (480, 640, 3)
        >>> gray = preprocessor.convert_to_grayscale(rgb_image)  # Shape: (480, 640)
        
        Notes:
        ------
        - OpenCV menggunakan BGR format by default (bukan RGB)
        - cv2.COLOR_BGR2GRAY sudah menggunakan formula optimal
        - Output range: 0 (hitam) sampai 255 (putih)
        """
        if image is None:
            raise ValueError("Input image is None.")
        
        # Cek apakah sudah grayscale
        if len(image.shape) == 2:
            if self.config.DEBUG_MODE:
                print("⚠️ Image sudah grayscale, skip conversion")
            return image.copy()
        
        # Cek jumlah channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR (OpenCV default) to Grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.config.DEBUG_MODE:
                print(f"✅ Grayscale: {image.shape} → {gray_image.shape}")
                print(f"   Min: {gray_image.min()}, Max: {gray_image.max()}, Mean: {gray_image.mean():.1f}")
            
            return gray_image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected (H, W, 3)")
    
    
    def preprocess(self, image):
        """
        Jalankan seluruh pipeline preprocessing.
        
        Pipeline:
        1. Resize image → (640, 480) atau sesuai aspect ratio
        2. Convert to grayscale → 1 channel
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (RGB/BGR)
            Shape: (height, width, 3)
        
        Returns:
        --------
        gray_image : numpy.ndarray
            Preprocessed image (grayscale)
            Shape: mendekati (480, 640)
        
        Example:
        --------
        >>> image = cv2.imread('jalan_berlubang.jpg')
        >>> preprocessed = preprocessor.preprocess(image)
        >>> cv2.imwrite('preprocessed.jpg', preprocessed)
        
        Notes:
        ------
        - Urutan penting: resize dulu, baru grayscale
        - Jika resize setelah grayscale, tidak ada perbedaan waktu signifikan
        - Tapi konvensi: resize dulu untuk konsistensi
        """
        if self.config.DEBUG_MODE:
            print("\n" + "="*60)
            print("🔧 PREPROCESSING PIPELINE")
            print("="*60)
        
        # Step 1: Resize
        resized = self.resize_image(image)
        
        # Step 2: Grayscale
        gray = self.convert_to_grayscale(resized)
        
        if self.config.DEBUG_MODE:
            print("="*60 + "\n")
        
        return gray


# ========== HELPER FUNCTIONS ==========

def load_image(image_path):
    """
    Load image dari file path.
    
    Parameters:
    -----------
    image_path : str
        Path ke file image
    
    Returns:
    --------
    image : numpy.ndarray or None
        Loaded image, atau None jika gagal load
    
    Example:
    --------
    >>> img = load_image('dataset/primary/damaged/foto1.jpg')
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
    return image


def save_image(image, output_path):
    """
    Simpan image ke file.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image yang akan disimpan
    output_path : str
        Path output file
    
    Returns:
    --------
    success : bool
        True jika berhasil simpan
    
    Example:
    --------
    >>> success = save_image(processed_img, 'output/result.jpg')
    """
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"✅ Image saved: {output_path}")
    else:
        print(f"❌ Failed to save image: {output_path}")
    return success


# ========== TEST CODE ==========

if __name__ == "__main__":
    """
    Test code untuk memverifikasi preprocessor bekerja dengan benar.
    """
    print("🧪 TESTING PREPROCESSOR MODULE")
    print("="*60)
    
    # Create dummy image untuk testing (simulasi foto 4000x3000)
    print("\n1️⃣ Creating dummy image...")
    dummy_image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    print(f"   Dummy image shape: {dummy_image.shape}")
    
    # Initialize preprocessor
    print("\n2️⃣ Initializing preprocessor...")
    config = Config()
    config.DEBUG_MODE = True  # Enable debug untuk melihat detail
    preprocessor = Preprocessor(config)
    print("   Preprocessor initialized ✅")
    
    # Test resize
    print("\n3️⃣ Testing resize...")
    resized = preprocessor.resize_image(dummy_image)
    print(f"   Resized shape: {resized.shape}")
    
    # Test grayscale
    print("\n4️⃣ Testing grayscale conversion...")
    gray = preprocessor.convert_to_grayscale(resized)
    print(f"   Grayscale shape: {gray.shape}")
    
    # Test full pipeline
    print("\n5️⃣ Testing full preprocessing pipeline...")
    processed = preprocessor.preprocess(dummy_image)
    print(f"   Final output shape: {processed.shape}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)