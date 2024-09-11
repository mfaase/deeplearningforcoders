from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str

model_path = './models/opus-mt-fr-en'
translator_tokenizer = MarianTokenizer.from_pretrained(model_path)
translator_model = MarianMTModel.from_pretrained(model_path)

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    src_text = [request.text]
    translated = translator_model.generate(**translator_tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return {"translated_text": translated_text}