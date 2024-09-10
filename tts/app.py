from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer
import torch
import io
import scipy.io.wavfile
from fastapi.responses import StreamingResponse

app = FastAPI()

# Define the input model for TTS
class TTSRequest(BaseModel):
    text: str

# Load the model and tokenizer from the local directory
model_path = "./models/mms-tts-eng/"
tts_model = VitsModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    # Validate input text
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required for TTS.")

    # Tokenize input text
    inputs = tokenizer(request.text, return_tensors="pt")
    
    # Move model and inputs to the correct device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate waveform (audio) from the model
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    # Convert the waveform tensor to numpy
    waveform_np = output[0].cpu().numpy()

    # Create a bytes buffer to store the audio data
    buffer = io.BytesIO()

    # Write the audio data to the buffer using scipy (assuming 16kHz sample rate)
    scipy.io.wavfile.write(buffer, rate=tts_model.config.sampling_rate, data=waveform_np)

    # Reset buffer position to the start
    buffer.seek(0)

    # Return the audio as a streaming response
    return StreamingResponse(buffer, media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output_audio.wav"
    })
