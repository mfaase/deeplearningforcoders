from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer
import torch
import io
import scipy.io.wavfile
from fastapi.responses import StreamingResponse

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

model_path = "./models/mms-tts-eng/"
tts_model = VitsModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    waveform_np = output[0].cpu().numpy()

    buffer = io.BytesIO()

    scipy.io.wavfile.write(buffer, rate=tts_model.config.sampling_rate, data=waveform_np)

    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output_audio.wav"
    })
