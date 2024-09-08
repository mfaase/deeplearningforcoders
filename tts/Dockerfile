FROM python:3.9-slim

# Install dependencies
RUN pip install fastapi uvicorn transformers torch scipy

# Copy model and app
COPY models /app/models
COPY app.py /app/app.py

WORKDIR /app

# Expose port for the TTS API
EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
