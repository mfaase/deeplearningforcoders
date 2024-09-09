from fastapi import FastAPI, UploadFile, File
from PIL import Image
import easyocr

app = FastAPI()

reader = easyocr.Reader(['fr'], gpu=False, model_storage_directory='./models/')

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    text_regions = reader.readtext(image, detail=1)
    text_result = " ".join([text[1] for text in text_regions])
    return {"text": text_result}