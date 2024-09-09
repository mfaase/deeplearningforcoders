import easyocr
from transformers import MarianMTModel, MarianTokenizer, VitsModel, AutoTokenizer

print("Downloading EasyOCR model...")
reader = easyocr.Reader(['fr'], gpu=False, model_storage_directory='./ocr/models/')

print("Downloading MarianMT model for translation...")
translation_model_dir = './translation/models/'
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", cache_dir=translation_model_dir)
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en", cache_dir=translation_model_dir)

print("Downloading Vits model for text-to-speech...")
tts_model_dir = './tts/models/'
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng", cache_dir=tts_model_dir)
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", cache_dir=tts_model_dir)

print("Models downloaded and stored successfully.")
