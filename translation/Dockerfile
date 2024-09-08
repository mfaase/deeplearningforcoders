FROM python:3.9-slim

# Install dependencies
RUN pip install fastapi uvicorn transformers torch

# Copy model and app
COPY models /app/models
COPY app.py /app/app.py

WORKDIR /app

# Expose port for the Translation API
EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
