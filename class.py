import os
import time
import uuid
import cv2
import traceback
import torch
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

# === Draw labeled image with class + confidence ===
def draw_labeled_image(image_path, particles, output_path):
    img = cv2.imread(image_path)

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

# === Fake particle classifier (replace with real model) ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def classify_particles(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Detect using YOLOv5
    results = model(img)
    df = results.pandas().xyxy[0]  # pandas DataFrame with detection info

    particles = []
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

    return particles

# === Main image processor loop ===
def process_images():
    while True:
        try:
            print("🔍 Checking for pending images...")
            response = supabase.table("images").select("*").eq("status", "pending").execute()
            images = response.data

            for image in images:
                image_id = image["id"]
                filename = image["filename"]
                path = os.path.join(UPLOAD_DIR, filename)
                print(f"🖼️ Processing: {filename}")

                try:
                    particles = classify_particles(path)

                    # Save labeled image
                    labeled_path = os.path.join(LABEL_DIR, filename)
                    draw_labeled_image(path, particles, labeled_path)

                    # Save particle crops and insert into DB
                    for i, part in enumerate(particles):
                        part_filename = f"{part['class']}_{image_id}_{i+1}.jpg"
                        particle_path = os.path.join(PARTICLE_DIR, part_filename)
                        cv2.imwrite(particle_path, part["crop"])

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
                    print(f"✅ Done: {filename}")

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
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(PARTICLE_DIR, exist_ok=True)
    process_images()
