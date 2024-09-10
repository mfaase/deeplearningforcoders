import easyocr
from transformers import MarianMTModel, MarianTokenizer, VitsModel, AutoTokenizer

# 1. Download and Save the EasyOCR model
print("Downloading EasyOCR model...")
reader = easyocr.Reader(['fr'], gpu=False, model_storage_directory='./ocr/models/')

# 2. Download and Save the MarianMT model for translation
print("Downloading MarianMT model for translation...")
translation_model_dir = './translation/models/opus-mt-fr-en'
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

# Save the tokenizer and model locally
translator_tokenizer.save_pretrained(translation_model_dir)
translator_model.save_pretrained(translation_model_dir)

print(f"MarianMT model and tokenizer saved to {translation_model_dir}")

# 3. Download and Save the Vits model for text-to-speech
print("Downloading Vits model for text-to-speech...")
tts_model_dir = './tts/models/mms-tts-eng'
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

# Save the tokenizer and model locally
tts_tokenizer.save_pretrained(tts_model_dir)
tts_model.save_pretrained(tts_model_dir)

print(f"Vits model and tokenizer saved to {tts_model_dir}")
