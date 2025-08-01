import cv2
import torch
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
from collections import defaultdict, deque
import threading
import queue

# ===== CONFIGURATION =====
CONFIG = {
    # Video source: 'camera', 'video', or 'image_folder'
    'SOURCE_TYPE': 'video',  # Change to 'video' or 'image_folder'
    
    # Camera settings (if SOURCE_TYPE = 'camera')
    'CAMERA_INDEX': 0,  # Usually 0 for default camera
    
    # Video file path (if SOURCE_TYPE = 'video')
    'VIDEO_PATH': 'ccvdo.mov',
    
    # Image folder path (if SOURCE_TYPE = 'image_folder')
    'IMAGE_FOLDER': 'input_images/',
    
    # Model settings
    'MODEL_PATH': 'best.pt',  # Path to your model.pt file
    'CONFIDENCE_THRESHOLD': 0.5,
    'NMS_THRESHOLD': 0.4,
    
    # API settings
    'API_BASE_URL': 'http://localhost:8080',  # Your Go API server
    'MACHINE_ID': '02e67550-1721-4e53-8f78-2d933a0e8725',  # Your machine ID
    
    # Detection settings
    'AUTO_UPLOAD': True,  # Automatically upload when detection occurs
    'UPLOAD_COOLDOWN': 2.0,  # Seconds between uploads
    'MAX_DETECTIONS_DISPLAY': 50,  # Max items in detection list
    
    # Display settings
    'WINDOW_WIDTH': 1280,
    'WINDOW_HEIGHT': 720,
    'SHOW_CONFIDENCE': True,
    'SHOW_CLASS_COUNTS': True,
}

class DetectionUploader:
    def __init__(self, api_base_url, machine_id):
        self.api_base_url = api_base_url
        self.machine_id = machine_id
        self.upload_queue = queue.Queue()
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        
    def _upload_worker(self):
        """Background thread to handle uploads"""
        while True:
            try:
                image_data, timestamp, detections = self.upload_queue.get(timeout=1)
                self._upload_image(image_data, timestamp, detections)
                self.upload_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Upload error: {e}")
    
    def _upload_image(self, image_data, timestamp, detections):
        """Upload image to server"""
        try:
            # Create temporary file
            temp_filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(temp_filename, image_data)
            
            # Prepare upload
            upload_url = f"{self.api_base_url}/api/upload"
            
            with open(temp_filename, 'rb') as f:
                files = {'image': (temp_filename, f, 'image/jpeg')}
                data = {'machine_id': self.machine_id}
                
                response = requests.post(upload_url, files=files, data=data, timeout=30)
                
                if response.status_code == 201:
                    result = response.json()
                    print(f"✅ Upload successful: {result.get('filename', 'unknown')}")
                    print(f"   Detections: {len(detections)} objects")
                else:
                    print(f"❌ Upload failed: {response.status_code} - {response.text}")
            
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        except Exception as e:
            print(f"❌ Upload exception: {e}")
    
    def queue_upload(self, image, detections):
        """Queue an image for upload"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        try:
            self.upload_queue.put_nowait((image.copy(), timestamp, detections))
            print(f"📤 Queued upload with {len(detections)} detections")
        except queue.Full:
            print("⚠️ Upload queue full, skipping...")

class RealtimeDetector:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.uploader = DetectionUploader(config['API_BASE_URL'], config['MACHINE_ID']) if config['AUTO_UPLOAD'] else None
        
        # Detection tracking
        self.detection_history = deque(maxlen=config['MAX_DETECTIONS_DISPLAY'])
        self.class_counts = defaultdict(int)
        self.last_upload_time = 0
        
        # Colors for different classes
        self.colors = {
            'alpha': (0, 0, 255),    # Red
            'beta': (0, 255, 255),   # Yellow  
            'muon': (0, 255, 0),     # Green
            'proton': (255, 0, 0),   # Blue
            'electron': (255, 0, 255), # Magenta
        }
        
    def _load_model(self):
        """Load the YOLO model"""
        try:
            print(f"🔧 Loading model from {self.config['MODEL_PATH']}...")
            
            # Try torch.hub YOLOv5 first (most compatible with existing models)
            try:
                print("🔧 Attempting to load with torch.hub YOLOv5...")
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path=self.config['MODEL_PATH'], force_reload=True)
                print("✅ Model loaded with torch.hub YOLOv5")
                return model
            except Exception as e:
                print(f"❌ torch.hub YOLOv5 failed: {e}")
            
            # Try with SSL fix for torch.hub
            try:
                print("🔧 Attempting torch.hub with SSL fix...")
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path=self.config['MODEL_PATH'], force_reload=True)
                print("✅ Model loaded with torch.hub + SSL fix")
                return model
            except Exception as e:
                print(f"❌ torch.hub with SSL fix failed: {e}")
            
            # Try loading from local cache (no internet required)
            try:
                print("🔧 Attempting to load from local cache...")
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path=self.config['MODEL_PATH'], force_reload=False)
                print("✅ Model loaded from local cache")
                return model
            except Exception as e:
                print(f"❌ Local cache failed: {e}")
            
            # Try loading from local YOLOv5 repository
            try:
                print("🔧 Attempting to load from local YOLOv5 repository...")
                model = torch.hub.load('./yolov5', 'custom', 
                                     path=self.config['MODEL_PATH'], source='local')
                print("✅ Model loaded from local YOLOv5 repository")
                return model
            except Exception as e:
                print(f"❌ Local YOLOv5 repository failed: {e}")
            
            # Last resort: try ultralytics (only for YOLOv8+ models)
            try:
                print("🔧 Last resort: trying ultralytics package (YOLOv8+)...")
                from ultralytics import YOLO
                model = YOLO(self.config['MODEL_PATH'])
                print("✅ Model loaded with ultralytics package")
                return model
            except Exception as e:
                print(f"❌ Ultralytics package failed: {e}")
                
            raise RuntimeError("All model loading methods failed. Please check your model file and internet connection.")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Suggestions:")
            print("   1. Make sure you have internet connection for torch.hub")
            print("   2. Try: pip install yolov5")
            print("   3. Clone YOLOv5 repo: git clone https://github.com/ultralytics/yolov5.git")
            raise
    
    def _get_video_source(self):
        """Get video source based on configuration"""
        if self.config['SOURCE_TYPE'] == 'camera':
            return cv2.VideoCapture(self.config['CAMERA_INDEX'])
        elif self.config['SOURCE_TYPE'] == 'video':
            return cv2.VideoCapture(self.config['VIDEO_PATH'])
        else:
            return None
    
    def _get_image_files(self):
        """Get list of image files from folder"""
        if self.config['SOURCE_TYPE'] != 'image_folder':
            return []
        
        folder = self.config['IMAGE_FOLDER']
        if not os.path.exists(folder):
            print(f"❌ Image folder not found: {folder}")
            return []
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
        return sorted([os.path.join(folder, f) for f in files])
    
    def _detect_objects(self, image):
        """Run detection on image"""
        detections = []
        
        try:
            if hasattr(self.model, 'predict'):  # Ultralytics YOLO
                results = self.model.predict(image, verbose=False, 
                                           conf=self.config['CONFIDENCE_THRESHOLD'],
                                           iou=self.config['NMS_THRESHOLD'])
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls.cpu().numpy()[0])
                            cls_name = self.model.names[cls_id]
                            conf = float(box.conf.cpu().numpy()[0])
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                            
                            detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'bbox': (x1, y1, x2, y2)
                            })
            else:  # torch.hub YOLOv5
                results = self.model(image)
                df = results.pandas().xyxy[0]
                
                for _, row in df.iterrows():
                    if row['confidence'] >= self.config['CONFIDENCE_THRESHOLD']:
                        detections.append({
                            'class': row['name'],
                            'confidence': float(row['confidence']),
                            'bbox': (int(row['xmin']), int(row['ymin']), 
                                   int(row['xmax']), int(row['ymax']))
                        })
        
        except Exception as e:
            print(f"❌ Detection error: {e}")
        
        return detections
    
    def _draw_detections(self, image, detections):
        """Draw detection boxes and labels on image"""
        for det in detections:
            cls_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Get color for class
            color = self.colors.get(cls_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if self.config['SHOW_CONFIDENCE']:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = cls_name
                
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return image
    
    def _draw_info_panel(self, image, detections):
        """Draw information panel on image"""
        h, w = image.shape[:2]
        panel_width = 300
        panel_height = min(400, h)
        
        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw title
        cv2.putText(image, "DETECTION STATUS", (w - panel_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        
        # Current detections
        cv2.putText(image, f"Current: {len(detections)}", (w - panel_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
        
        # Class counts
        if self.config['SHOW_CLASS_COUNTS'] and self.class_counts:
            cv2.putText(image, "Total Counts:", (w - panel_width + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
            
            for cls_name, count in sorted(self.class_counts.items()):
                color = self.colors.get(cls_name, (255, 255, 255))
                cv2.putText(image, f"  {cls_name}: {count}", 
                           (w - panel_width + 20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 18
        
        # Recent detections
        if self.detection_history:
            y_offset += 10
            cv2.putText(image, "Recent Detections:", (w - panel_width + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
            
            for i, (timestamp, det_count, classes) in enumerate(list(self.detection_history)[-10:]):
                time_str = timestamp.strftime("%H:%M:%S")
                text = f"{time_str}: {det_count} ({', '.join(classes)})"
                cv2.putText(image, text[:35], (w - panel_width + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                y_offset += 15
        
        # Instructions
        y_offset = h - 60
        cv2.putText(image, "Controls:", (w - panel_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 15
        cv2.putText(image, "SPACE: Manual upload", (w - panel_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        y_offset += 12
        cv2.putText(image, "ESC/Q: Quit", (w - panel_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return image
    
    def _update_detection_history(self, detections):
        """Update detection history and counts"""
        if detections:
            timestamp = datetime.now()
            classes = [det['class'] for det in detections]
            
            # Update total counts
            for cls in classes:
                self.class_counts[cls] += 1
            
            # Add to history
            self.detection_history.append((timestamp, len(detections), classes))
    
    def _should_upload(self, detections):
        """Check if should upload based on detections and cooldown"""
        if not self.config['AUTO_UPLOAD'] or not detections:
            return False
        
        current_time = time.time()
        if current_time - self.last_upload_time < self.config['UPLOAD_COOLDOWN']:
            return False
        
        return True
    
    def run(self):
        """Main detection loop"""
        print(f"🚀 Starting detection with source: {self.config['SOURCE_TYPE']}")
        
        if self.config['SOURCE_TYPE'] in ['camera', 'video']:
            self._run_video_detection()
        elif self.config['SOURCE_TYPE'] == 'image_folder':
            self._run_image_folder_detection()
        else:
            print("❌ Invalid source type")
    
    def _run_video_detection(self):
        """Run detection on video/camera stream"""
        cap = self._get_video_source()
        
        if not cap.isOpened():
            print("❌ Failed to open video source")
            return
        
        print("✅ Video source opened successfully")
        print("Press SPACE to manually upload current frame")
        print("Press ESC or Q to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.config['SOURCE_TYPE'] == 'video':
                        print("📹 Video ended")
                        break
                    else:
                        print("❌ Failed to read from camera")
                        continue
                
                # Resize frame if needed
                if frame.shape[1] > self.config['WINDOW_WIDTH']:
                    aspect_ratio = frame.shape[0] / frame.shape[1]
                    new_width = self.config['WINDOW_WIDTH']
                    new_height = int(new_width * aspect_ratio)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Run detection
                detections = self._detect_objects(frame)
                
                # Update history
                self._update_detection_history(detections)
                
                # Auto upload if conditions met
                if self._should_upload(detections):
                    self.uploader.queue_upload(frame, detections)
                    self.last_upload_time = time.time()
                
                # Draw results
                display_frame = self._draw_detections(frame.copy(), detections)
                display_frame = self._draw_info_panel(display_frame, detections)
                
                # Show frame
                cv2.imshow('Particle Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE for manual upload
                    if self.uploader and detections:
                        self.uploader.queue_upload(frame, detections)
                        print("📤 Manual upload triggered")
                    else:
                        print("⚠️ No detections to upload")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _run_image_folder_detection(self):
        """Run detection on images from folder"""
        image_files = self._get_image_files()
        
        if not image_files:
            print("❌ No images found in folder")
            return
        
        print(f"📁 Found {len(image_files)} images")
        print("Press SPACE to go to next image")
        print("Press ESC or Q to quit")
        
        for i, image_path in enumerate(image_files):
            print(f"🖼️ Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ Failed to load {image_path}")
                continue
            
            # Resize if needed
            if frame.shape[1] > self.config['WINDOW_WIDTH']:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_width = self.config['WINDOW_WIDTH']
                new_height = int(new_width * aspect_ratio)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Run detection
            detections = self._detect_objects(frame)
            
            # Update history
            self._update_detection_history(detections)
            
            # Auto upload if conditions met
            if self._should_upload(detections):
                self.uploader.queue_upload(frame, detections)
                self.last_upload_time = time.time()
            
            # Draw results
            display_frame = self._draw_detections(frame.copy(), detections)
            display_frame = self._draw_info_panel(display_frame, detections)
            
            # Show frame
            cv2.imshow('Particle Detection', display_frame)
            
            # Wait for key press
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):  # SPACE for next image
                    break
                elif key == ord('u'):  # U for manual upload
                    if self.uploader and detections:
                        self.uploader.queue_upload(frame, detections)
                        print("📤 Manual upload triggered")
        
        cv2.destroyAllWindows()

def main():
    # Validate configuration
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print(f"❌ Model file not found: {CONFIG['MODEL_PATH']}")
        return
    
    if CONFIG['SOURCE_TYPE'] == 'video' and not os.path.exists(CONFIG['VIDEO_PATH']):
        print(f"❌ Video file not found: {CONFIG['VIDEO_PATH']}")
        return
    
    if CONFIG['SOURCE_TYPE'] == 'image_folder' and not os.path.exists(CONFIG['IMAGE_FOLDER']):
        print(f"❌ Image folder not found: {CONFIG['IMAGE_FOLDER']}")
        return
    
    # Create and run detector
    detector = RealtimeDetector(CONFIG)
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()