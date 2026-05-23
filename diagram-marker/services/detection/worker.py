import os
import json
import logging
import time
from confluent_kafka import Consumer, Producer
import requests
from io import BytesIO
from PIL import Image
try:
    from ultralytics import YOLO
    import torch
    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'detection_group',
    'auto.offset.reset': 'earliest'
})
producer = Producer({'bootstrap.servers': KAFKA_BROKER})

# Initialize YOLO model (will use CPU if GPU is unavailable)
# In production, this would be a custom trained weight file e.g., 'weights/diagram_v1.pt'
model_path = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt") 

if HAS_ML:
    logger.info(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.warning(f"Failed to load YOLO weights: {e}. Running in graceful degradation mode.")
        HAS_ML = False

def run_yolo_inference(image_url):
    """
    Downloads image, runs YOLOv8 inference, returns structured boxes.
    Classes expected: 0=region, 1=arrow, 2=label
    """
    if not HAS_ML:
        logger.warning("ML libraries missing. Using mock inference.")
        time.sleep(0.5)
        return [
            {"cls": 2, "label": "label", "text": "aorta", "bbox": [10, 20, 50, 60], "conf": 0.95, "center": [30, 40]},
            {"cls": 1, "label": "arrow", "bbox": [55, 40, 80, 45], "conf": 0.88, "center": [67, 42]},
            {"cls": 0, "label": "region", "bbox": [85, 20, 150, 80], "conf": 0.92, "center": [117, 50]}
        ]

    # Real Inference Logic
    logger.info("Running real YOLOv8 inference pass...")
    try:
        if image_url.startswith('http'):
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            # Local file path for testing
            img = Image.open(image_url)
    except Exception as e:
        logger.error(f"Failed to load image from {image_url}: {e}")
        return []

    results = model(img)
    boxes = results[0].boxes
    
    extracted = []
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        
        # Determine label text (usually 0=region, 1=arrow, 2=label)
        cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)
        
        extracted.append({
            "cls": cls_id, 
            "label": cls_name,
            "bbox": xyxy,
            "conf": conf,
            "center": [cx, cy]
        })
    return extracted


def main():
    consumer.subscribe(['diagram.raw.uploaded'])
    logger.info("Detection Worker started.")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error(): continue

            payload = json.loads(msg.value().decode('utf-8'))
            task_id = payload.get("task_id")
            logger.info(f"YOLO Inference processing task {task_id}")

            regions = run_yolo_inference(payload.get("image_url"))
            
            out_msg = {
                "task_id": task_id,
                "image_url": payload.get("image_url"), # Pass URL forward for OCR
                "diagram_crops": regions,
                "rubric": payload.get("rubric")
            }
            producer.produce('diagram.regions.extracted', key=task_id, value=json.dumps(out_msg).encode('utf-8'))
            producer.poll(0)
            
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
