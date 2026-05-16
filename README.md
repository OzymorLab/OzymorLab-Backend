# Edexia Backend: AI Assessment Operating System (AIOS)

Welcome to the **Edexia AIOS Backend**, the core orchestration layer for processing, parsing, and evaluating handwritten student answer sheets at a national scale.

## What is this repository?
This is the **Core Monolith**. It acts as the brain of the Edexia Assessment Engine. It handles everything from student file uploads to the final LLM-driven grading justification. It coordinates parsing, dynamic rubric injection, mathematical equation verification, and orchestrates async evaluation workflows using Celery and Redis.

### Core Features:
- **Stateful Document Parsing**: Doesn't just blindly read text. It uses a custom Context Memory Tracker to understand "implicit human continuity" across pages (e.g., inferring a student is still answering Section B, Question 4b even if they forgot to write the header).
- **Gemini Vision OCR**: Uses Gemini 2.5 Pro Vision to bypass the brittleness of traditional Tesseract OCR, enabling hyper-accurate parsing of messy, handwritten Indian board exam answers.
- **Dynamic Knowledge Base**: Automatically injects exact grading guidelines and board-specific rulesets (e.g., CBSE Class 10 Science vs ICSE Class 12 Math) dynamically via Markdown files.
- **SymPy Mathematical Validation**: A deterministic secondary pass that computationally verifies mathematical derivations before the LLM grades them, preventing LLM arithmetic hallucination.
- **AWS S3 Storage Pipeline**: Handles high-throughput document ingestion.

## Tech Stack
- **Framework**: FastAPI (Python 3.11)
- **Database**: PostgreSQL (SQLAlchemy + Alembic)
- **Queuing**: Celery & Redis
- **AI Core**: Google Gemini 2.5 Pro (Multimodal)
- **Storage**: AWS S3 via boto3

## How it works with the broader system
1. The **Frontend Dashboard** sends student PDF submissions here.
2. The AIOS Backend extracts the text, segments the logic steps, and evaluates the text/math.
3. If an educational diagram (like a Physics Circuit or Human Heart) is detected, it is shunted over to the **DEIS (Diagram-Marker)** microservice cluster for heavy GPU evaluation.