from fastapi import FastAPI
from transformers import VitsModel, AutoTokenizer
import torch
import scipy

app = FastAPI()

model_path = "./models/facebook/"
tts_model = VitsModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.post("/tts/")
async def text_to_speech(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    scipy.io.wavfile.write("/app/output_audio.wav", rate=tts_model.config.sampling_rate, data=output[0].float().numpy())
    return {"message": "Audio generated", "path": "/app/output_audio.wav"}
