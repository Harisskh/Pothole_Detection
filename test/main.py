import cv2
import numpy as np

class PotholeDetector:
    """
    Sistem deteksi lubang jalan menggunakan Canny Edge Detection
    Support: Foto (single image) dan Video (sequence of images)
    """
    
    def __init__(self):
        self.min_size = 15
        self.max_size = 290
    
    # ========== CORE IMAGE PROCESSING ==========
    def detect_single_image(self, image):
        """Deteksi lubang pada 1 foto"""
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(thresh, 0, 150)
        
        # 4. Dilasi
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 5. Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. Filter berdasarkan ukuran
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (self.min_size <= w <= self.max_size and 
                self.min_size <= h <= self.max_size):
                boxes.append((x, y, w, h))
        
        return boxes
    
    # ========== VIDEO SUPPORT ==========
    def detect_video(self, video_path, fps_sample=1):
        """
        Deteksi lubang pada video (sequence of images)
        
        fps_sample: berapa frame per detik yang diambil
                   1 = 1 frame/detik, 0.5 = 1 frame/2 detik
        """
        # Step 1: Extract frames
        frames = self._extract_frames(video_path, fps_sample)
        print(f"Extracted {len(frames)} frames")
        
        # Step 2: Detect per frame (IMAGE PROCESSING)
        all_detections = []
        for i, frame in enumerate(frames):
            boxes = self.detect_single_image(frame)
            for box in boxes:
                all_detections.append({
                    'frame_id': i,
                    'box': box,
                    'image': frame
                })
        
        print(f"Total detections: {len(all_detections)}")
        
        # Step 3: Deduplication
        unique = self._deduplicate(all_detections)
        print(f"Unique potholes: {len(unique)}")
        
        return unique
    
    def _extract_frames(self, video_path, fps_sample):
        """Extract frames dari video"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps_sample)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize untuk efisiensi
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _deduplicate(self, detections, threshold=50):
        """Hapus deteksi duplikat"""
        unique = []
        
        for det in detections:
            x, y, w, h = det['box']
            center = (x + w/2, y + h/2)
            
            is_duplicate = False
            for u in unique:
                u_x, u_y, u_w, u_h = u['box']
                u_center = (u_x + u_w/2, u_y + u_h/2)
                
                # Hitung jarak
                dist = np.sqrt((center[0] - u_center[0])**2 + 
                              (center[1] - u_center[1])**2)
                
                if dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(det)
        
        return unique
    
    # ========== VISUALIZATION ==========
    def draw_results(self, image, boxes):
        """Gambar kotak merah di lubang"""
        result = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return result


# ========== CARA PAKAI ==========
if __name__ == "__main__":
    detector = PotholeDetector()
    
    # TEST 1: Single image
    print("=== Testing with photo ===")
    img = cv2.imread('jalan_foto.jpg')
    boxes = detector.detect_single_image(img)
    result = detector.draw_results(img, boxes)
    cv2.imwrite('hasil_foto.jpg', result)
    print(f"Detected {len(boxes)} potholes in photo")
    
    # TEST 2: Video
    print("\n=== Testing with video ===")
    unique_potholes = detector.detect_video('jalan_video.mp4', fps_sample=1)
    
    # Simpan hasil
    for i, pothole in enumerate(unique_potholes):
        img_with_box = detector.draw_results(pothole['image'], 
                                             [pothole['box']])
        cv2.imwrite(f'hasil_video_lubang_{i+1}.jpg', img_with_box)
    
    print(f"Saved {len(unique_potholes)} unique potholes")
