import os
import time
import uuid
import cv2
import traceback
import torch
import ssl
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client

# === Load environment variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 30))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "upload")
LABEL_DIR = os.getenv("LABEL_DIR", "label")
PARTICLE_DIR = os.getenv("PARTICLE_DIR", "particle")

# === Supabase Init ===
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Load model with SSL fix ===
def load_model():
    try:
        # Method 1: Try loading with SSL certificate fix
        print("🔧 Loading model with SSL fix...")
        ssl._create_default_https_context = ssl._create_unverified_context
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        print("✅ Model loaded successfully with SSL fix")
        return model
    except Exception as e:
        print(f"❌ SSL fix failed: {e}")
        
        try:
            # Method 2: Try loading from local cache (no internet required)
            print("🔧 Trying to load from local cache...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
            print("✅ Model loaded from local cache")
            return model
        except Exception as e2:
            print(f"❌ Local cache failed: {e2}")
            
            try:
                # Method 3: Try loading with local YOLOv5 repo
                print("🔧 Trying to load from local YOLOv5 repository...")
                model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')
                print("✅ Model loaded from local repository")
                return model
            except Exception as e3:
                print(f"❌ Local repository failed: {e3}")
                
                try:
                    # Method 4: Try using ultralytics package (if installed)
                    print("🔧 Trying ultralytics package...")
                    from ultralytics import YOLO
                    model = YOLO('best.pt')
                    print("✅ Model loaded with ultralytics package")
                    return model
                except Exception as e4:
                    print(f"❌ Ultralytics package failed: {e4}")
                    print("💡 Please install ultralytics: pip install ultralytics")
                    raise RuntimeError("All model loading methods failed. Please check your internet connection, SSL certificates, or install ultralytics package.")

# === Draw labeled image with class + confidence ===
def draw_labeled_image(image_path, particles, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    for part in particles:
        x, y, w, h = part["x"], part["y"], part["w"], part["h"]
        cls = part["class"]
        conf = part["confidence"]

        # Choose color per class
        if cls == "alpha":
            color = (0, 0, 255)   # Red (BGR)
        elif cls == "muon":
            color = (0, 255, 0)   # Green
        else:
            color = (0, 255, 255) # Yellow for beta

        # Draw box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{cls} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y), color, -1)
        cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imwrite(output_path, img)

# === Particle classifier ===
def classify_particles(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    particles = []
    
    try:
        # Check if it's ultralytics YOLO model
        if hasattr(model, 'predict'):
            # Ultralytics YOLO
            results = model.predict(img, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls.cpu().numpy()[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf.cpu().numpy()[0])
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        w, h = x2 - x1, y2 - y1
                        crop = img[y1:y2, x1:x2]
                        
                        particles.append({
                            "class": cls_name,
                            "confidence": conf,
                            "x": x1,
                            "y": y1,
                            "w": w,
                            "h": h,
                            "crop": crop
                        })
        else:
            # Traditional YOLOv5 with torch.hub
            results = model(img)
            df = results.pandas().xyxy[0]  # pandas DataFrame with detection info
            
            for i, row in df.iterrows():
                cls = row['name']
                conf = float(row['confidence'])
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                w, h = x2 - x1, y2 - y1
                crop = img[y1:y2, x1:x2]

                particles.append({
                    "class": cls,
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                    "crop": crop
                })
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        traceback.print_exc()
        raise

    return particles

# === Main image processor loop ===
def process_images():
    # Load model once at startup
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    while True:
        try:
            print("🔍 Checking for pending images...")
            response = supabase.table("images").select("*").eq("status", "pending").execute()
            images = response.data

            if not images:
                print("📭 No pending images found")
            else:
                print(f"📧 Found {len(images)} pending images")

            for image in images:
                image_id = image["id"]
                filename = image["filename"]
                path = os.path.join(UPLOAD_DIR, filename)
                print(f"🖼️ Processing: {filename}")

                try:
                    # Check if file exists
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Image file not found: {path}")
                    
                    particles = classify_particles(path, model)
                    print(f"🔍 Found {len(particles)} particles")

                    # Save labeled image
                    labeled_path = os.path.join(LABEL_DIR, filename)
                    draw_labeled_image(path, particles, labeled_path)

                    # Save particle crops and insert into DB
                    for i, part in enumerate(particles):
                        part_filename = f"{part['class']}_{image_id}_{i+1}.jpg"
                        particle_path = os.path.join(PARTICLE_DIR, part_filename)
                        
                        # Ensure crop is valid before saving
                        if part["crop"].size > 0:
                            cv2.imwrite(particle_path, part["crop"])
                        else:
                            print(f"⚠️ Skipping empty crop for particle {i+1}")
                            continue

                        supabase.table("particles").insert({
                            "id": str(uuid.uuid4()),
                            "image_id": image_id,
                            "particle_filename": part_filename,
                            "class": part["class"],
                            "confidence": part["confidence"],
                            "x_position": part["x"],
                            "y_position": part["y"],
                            "width": part["w"],
                            "height": part["h"]
                        }).execute()

                    supabase.table("images").update({"status": "detected"}).eq("id", image_id).execute()
                    print(f"✅ Done: {filename} - {len(particles)} particles detected")

                except Exception as e:
                    print(f"❌ Error on {filename}: {e}")
                    traceback.print_exc()
                    supabase.table("images").update({"status": "fail"}).eq("id", image_id).execute()

        except Exception as global_err:
            print(f"⚠️ Global error: {global_err}")
            traceback.print_exc()

        time.sleep(CHECK_INTERVAL)

# === Startup ===
if __name__ == "__main__":
    print("🚀 Starting Particle Detection System...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Label directory: {LABEL_DIR}")
    print(f"📁 Particle directory: {PARTICLE_DIR}")
    print(f"⏱️ Check interval: {CHECK_INTERVAL}s")
    
    # Create directories
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(PARTICLE_DIR, exist_ok=True)
    
    # Start processing
    process_images()