import os
import json
import uuid
import logging
import redis
from fastapi import FastAPI, HTTPException
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DEIS API Gateway")

# Configure Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Configure Kafka Producer
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
producer = Producer({"bootstrap.servers": KAFKA_BROKER})

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

@app.post("/api/v1/diagram/evaluate")
async def evaluate_diagram(payload: dict):
    """
    Ingests diagram evaluation request and pushes to Kafka.

    Expected payload:
    {
        "image_url": "https://...",
        "question_id": "q123",
        "submission_id": "uuid",  # Edexia submission ID for traceability
        "step_num": 3,            # Rubric step number this diagram belongs to
        "rubric": {
            "max_marks": 5,
            "relations": [{"label": "aorta", "region": "region_0"}, ...]
        }
    }
    """
    task_id = str(uuid.uuid4())
    
    # Store initial status in Redis
    redis_client.set(f"diagram_result:{task_id}", json.dumps({
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Queued for GPU detection",
        "submission_id": payload.get("submission_id", ""),
        "step_num": payload.get("step_num", 0),
    }), ex=86400) # 24h TTL

    # Publish to Kafka
    kafka_msg = {
        "task_id": task_id,
        "image_url": payload.get("image_url", ""),
        "question_id": payload.get("question_id", ""),
        "submission_id": payload.get("submission_id", ""),
        "step_num": payload.get("step_num", 0),
        "rubric": payload.get("rubric", {})
    }
    
    try:
        producer.produce(
            "diagram.raw.uploaded",
            key=task_id,
            value=json.dumps(kafka_msg).encode('utf-8'),
            callback=delivery_report
        )
        producer.poll(0) # Trigger callbacks
    except Exception as e:
        logger.error(f"Failed to publish to Kafka: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue task")

    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Diagram successfully queued for GPU analysis"
    }

@app.get("/api/v1/diagram/status/{task_id}")
async def get_status(task_id: str):
    """
    Polls Redis for the evaluation result.
    """
    result = redis_client.get(f"diagram_result:{task_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return json.loads(result)
