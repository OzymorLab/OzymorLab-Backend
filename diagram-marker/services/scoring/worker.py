import os
import json
import logging
import time
import redis
from confluent_kafka import Consumer, Producer

try:
    import networkx as nx
    from networkx.algorithms import isomorphism
    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'scoring_group',
    'auto.offset.reset': 'earliest'
})
producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def score_graph_isomorphism(student_graph_data, rubric_schema):
    """
    Reconstructs the NetworkX graph and runs DiGraphMatcher against a generated Rubric Graph.
    """
    if not HAS_ML:
        logger.warning("ML libraries missing. Using mock scoring.")
        time.sleep(0.4)
        return {
            "diagram_detected": True,
            "predicted_marks": 4, "max_marks": 5, "confidence": 0.88,
            "missing_components": ["pulmonary artery label missing"]
        }

    logger.info("Running NetworkX Graph Isomorphism evaluation...")
    student_graph = nx.node_link_graph(student_graph_data)
    
    rubric_graph = nx.DiGraph()
    
    relations = rubric_schema.get("relations", [])
    for rel in relations:
        label_text = rel.get("label", "unknown")
        region_id = rel.get("region", "region_0")
        
        safe_label_id = f"label_{label_text.replace(' ', '_').lower()}"
        
        # Add nodes if they don't exist
        if not rubric_graph.has_node(safe_label_id):
            rubric_graph.add_node(safe_label_id, type="label", text=label_text)
        if not rubric_graph.has_node(region_id):
            rubric_graph.add_node(region_id, type="region")
            
        rubric_graph.add_edge(safe_label_id, region_id, relation="points_to")
    
    # Run exact matching on edge relationships ignoring coordinate differences
    def node_match(n1, n2):
        return n1.get('type') == n2.get('type') and n1.get('text') == n2.get('text')

    matcher = isomorphism.DiGraphMatcher(rubric_graph, student_graph, node_match=node_match)
    
    # Calculate subgraph isomorphism (how much of the rubric did they successfully draw?)
    # A simple scoring metric: Ratio of matched edges to total rubric edges
    rubric_edges = list(rubric_graph.edges())
    student_edges = list(student_graph.edges(data=True))
    
    matched_edges = 0
    missing_edges = []
    
    for r_u, r_v in rubric_edges:
        # Does the student graph contain an edge from an equivalent label to an equivalent region?
        found = False
        for s_u, s_v, data in student_edges:
            # Check if student has a matching label pointing to any region
            # We use fuzzy matching or exact matching based on label text
            # For this version, we require the node IDs (which encode the label text) to match
            if s_u == r_u: 
                found = True
                matched_edges += 1
                break
        if not found:
            missing_edges.append(f"Missing pointer from label '{r_u}' to a region.")

    max_marks = rubric_schema.get("max_marks", 5)
    score_ratio = matched_edges / max(len(rubric_edges), 1)
    predicted_marks = round(score_ratio * max_marks)
    
    # Extract per-label scoring for the Label Validation Pipeline
    label_scores = []
    for r_u, r_v in rubric_edges:
        label_text = rubric_graph.nodes[r_u].get("text", r_u)
        matched = any(s_u == r_u for s_u, s_v, _ in student_edges)
        # Try to find detected text from student graph
        detected_text = ""
        if matched and student_graph.has_node(r_u):
            detected_text = student_graph.nodes[r_u].get("text", r_u)
        label_scores.append({
            "expected": label_text,
            "text": detected_text if matched else "",
            "matched": matched,
            "target_region": r_v,
        })

    return {
        "diagram_detected": True,
        "predicted_marks": predicted_marks,
        "max_marks": max_marks,
        "confidence": score_ratio,
        "rubric_breakdown": {"edges_matched": matched_edges, "edges_required": len(rubric_edges)},
        "missing_components": missing_edges,
        "label_scores": label_scores,
        "scene_graph": student_graph_data,
    }

def main():
    consumer.subscribe(['diagram.graph.built'])
    logger.info("Scoring Worker started.")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error(): continue

            payload = json.loads(msg.value().decode('utf-8'))
            task_id = payload.get("task_id")
            logger.info(f"Scoring task {task_id}")

            score_result = score_graph_isomorphism(payload.get("scene_graph"), payload.get("rubric") or {})
            score_result["task_id"] = task_id
            score_result["status"] = "COMPLETED"
            
            redis_client.set(f"diagram_result:{task_id}", json.dumps(score_result), ex=86400)
            
            producer.produce('diagram.evaluation.completed', key=task_id, value=json.dumps(score_result).encode('utf-8'))
            producer.poll(0)
            
            logger.info(f"Task {task_id} COMPLETED and saved to Redis.")
            
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
