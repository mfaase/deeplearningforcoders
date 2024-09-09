from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

model_path = "./models/"
translator_tokenizer = MarianTokenizer.from_pretrained(model_path)
translator_model = MarianMTModel.from_pretrained(model_path)

@app.post("/translate/")
async def translate_text(text: str):
    src_text = [text]
    translated = translator_model.generate(**translator_tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return {"translated_text": translated_text}
