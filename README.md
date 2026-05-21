# Edexia Backend: AI Assessment Operating System (AIOS)

Welcome to the **Edexia AIOS Backend**, the core orchestration layer for processing, parsing, and evaluating handwritten student answer sheets at an institutional scale.

## What is this repository?
This is the **Core Monolith**. It acts as the brain of the Edexia Assessment Engine. It handles everything from student file uploads to the final LLM-driven grading justification. It coordinates parsing, dynamic rubric injection, mathematical equation verification, and orchestrates async evaluation workflows using Celery and Redis.

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
- **Queuing**: Celery & Redis
- **AI Core**: Google Gemini 2.5 Pro (Multimodal)
- **Authentication & Storage**: Supabase Auth + Supabase Storage

## CI/CD Pipeline
The repository uses GitHub Actions for continuous integration. On every push and pull request to active branches, the pipeline automatically:
1. Provisions a `postgres:15` service container.
2. Applies all Alembic database migrations.
3. Injects mock environment variables (e.g., Mock Supabase URLs, Mock Gemini Keys).
4. Executes the full `PyTest` suite and `Flake8` linting to prevent regressions.

## How it works with the broader system
1. The **Frontend Dashboard** sends student PDF submissions here, fully authenticated via Supabase JWTs.
2. The AIOS Backend extracts the text, segments the logic steps, and evaluates the text/math concurrently.
3. If an educational diagram (like a Physics Circuit or Human Heart) is detected, it is shunted over to the **DEIS (Diagram-Marker)** microservice cluster for heavy GPU evaluation.
4. Evaluation results are aggregated using a robust Score Fusion engine and presented back to the teacher for final moderation.