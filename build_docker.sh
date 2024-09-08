#!/bin/bash

# Build OCR Docker image
docker build -t ocr-service ./ocr

# Build Translation Docker image
docker build -t translation-service ./translation

# Build TTS Docker image
docker build -t tts-service ./tts
