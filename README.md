# OzymorLab: AI Assessment Operating System (AIOS)

Welcome to the **OzymorLab**, the core orchestration layer for processing, parsing, and evaluating handwritten student answer sheets at an institutional scale.

## What is this repository?
This is the **Core Monolith**. It acts as the brain of the Assessment Engine. It handles everything from student file uploads to the final LLM-driven grading justification. It coordinates parsing, dynamic rubric injection, mathematical equation verification, and orchestrates async evaluation workflows using FastAPI `BackgroundTasks` and `asyncio`.

### Core Features:
- **Multi-Tenant Institutional Architecture**: Built to support large-scale rollouts with robust hierarchical models mapping `Schools`, `ExamCycles`, `Class Standards`, `Sections`, and `Students`. It enforces strict multi-level Role-Based Access Control (RBAC) across Teachers, Evaluators, HODs, and Principals.
- **Stateful Document Parsing**: Doesn't just blindly read text. It uses a custom Context Memory Tracker to understand "implicit human continuity" across pages (e.g., inferring a student is still answering Section B, Question 4b even if they forgot to write the header).
- **Gemini Vision OCR**: Uses Gemini 2.5 Pro Vision to bypass the brittleness of traditional Tesseract OCR, enabling hyper-accurate parsing of messy, handwritten Indian board exam answers. It securely manages BYOK (Bring Your Own Key) capabilities through Fernet symmetric encryption.
- **Dynamic Knowledge Base**: Automatically injects exact grading guidelines and board-specific rulesets (e.g., CBSE Class 10 Science vs ICSE Class 12 Math) dynamically via Markdown files.
- **SymPy Mathematical Validation**: A deterministic secondary pass that computationally verifies mathematical derivations before the LLM grades them, preventing LLM arithmetic hallucination.
- **Supabase Storage Pipeline**: Replaces legacy systems to handle high-throughput, secure document ingestion using authenticated REST endpoints.

## Tech Stack
- **Framework**: FastAPI (Python 3.12)
- **Database**: PostgreSQL (SQLAlchemy + Asyncpg + Alembic)
- **Background Tasks**: FastAPI BackgroundTasks + asyncio (no external broker needed)
- **AI Core**: Google Gemini 2.5 Pro (Multimodal)
- **Authentication & Storage**: Supabase Auth + Supabase Storage

## CI/CD Pipeline
The repository uses GitHub Actions for continuous integration. On every push and pull request to active branches, the pipeline automatically:
1. Provisions a `postgres:15` service container.
2. Applies all Alembic database migrations.
3. Injects mock environment variables (e.g., Mock Supabase URLs, Mock Gemini Keys).
4. Executes the full `PyTest` suite and `Flake8` linting to prevent regressions.

## Integrated Diagram Evaluation System (DEIS)
The **Diagram Evaluation Intelligence System (DEIS)** is consolidated under `/diagram-marker` and runs as a microservice cluster inside the same Docker Compose network:
- **`deis-gateway`**: Exposes a fast HTTP polling endpoint (`http://deis-gateway:8001/api/v1/diagram/evaluate`) for submitting images.
- **`deis-detection`**: A worker loading **YOLOv8** weights to extract bounding boxes for labels, arrows, and regions.
- **`deis-structural`**: Analyzes arrow-label-region geometries and constructs a directed scene graph.
- **`deis-scoring`**: Deterministically compares the student's mathematical scene graph against the golden rubric graph using `DiGraphMatcher` for partial scoring.

All services communicate asynchronously using an **Apache Kafka** broker and cache intermediate states/results in the shared **Redis** container.

## Running the Unified Stack

To start all components (PostgreSQL, AIOS Web Server, Kafka, Zookeeper, and all DEIS microservices):

```bash
docker compose up --build
```

You can verify the components are running:
- Main Backend API: `http://localhost:8000/docs`
- DEIS Gateway API: `http://localhost:8001/docs`
