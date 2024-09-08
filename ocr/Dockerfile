FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi uvicorn easyocr pillow

# Copy model and app
COPY models /app/models
COPY app.py /app/app.py

WORKDIR /app

# Expose port for the OCR API
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
