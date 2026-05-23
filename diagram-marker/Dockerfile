FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code into container
COPY . .

# Start the gateway service using uvicorn binding to Render's dynamic PORT
CMD ["sh", "-c", "uvicorn services.gateway.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
