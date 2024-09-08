#!/bin/bash

# Run the OCR service container
docker run -d --name ocr-service -p 8000:8000 ocr-service

# Run the Translation service container
docker run -d --name translation-service -p 8001:8001 translation-service

# Run the TTS service container
docker run -d --name tts-service -p 8002:8002 tts-service
