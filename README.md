Building Deep Learning Apps for Coders:
- 
This project demonstrates how to build and deploy three key deep learning models: OCR (Optical Character Recognition), Translation, and TTS (Text-to-Speech)—as microservices using Docker and FastAPI. These models are integrated into an inferencing pipeline that can be tested using a Jupyter notebook.

Installation Requirements:
-
Install the following dependencies:
* **(mini)conda**: A package and environment manager for Python that simplifies the management of dependencies
* **some unix distro**: the scripts are designed for unix-like environments
* **Docker engine**: to containerize and run the microservices in isolated environments
* **git**: for cloning the repository: 
* **useful**: VSCode with Jupyter Extension enabled for the inferencing notebook for testing


Walkthrough:
-
Follow the steps below to set up the project and run the inferencing pipeline:

### 1. Clone the repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/mfaase/deeplearningforcoders
cd deeplearningforcoders
```
After cloning the repository, move into the project directory.

### 2. Create and activate the Conda environment
Set up the Conda environment from the `requirements.yml` file:
```bash
conda env create -f requirements.yml
```
This command creates a Conda environment named ocr_translation_tts_env with all the required Python packages (e.g., FastAPI, transformers, torch, etc.).

Verify the environment was created by listing available environments:
```bash
conda info -e
```
Then activate the environment so python runs in the context of that environment:
```bash
conda activate ocr_translation_tts_env
```

### 3. Download pre-trained models
To ensure that the microservices (OCR, Translation, TTS) have access to the pre-trained models, run the following script to download and save them locally:
```bash
python download_models.py
```
This script downloads the OCR, MarianMT translation, and VITS TTS models from Hugging Face and stores them in the correct folder structure.

You can find the models and their documentation here: 
- Optical Character Recognition (OCR): [JaidedAI/easyocr](https://www.jaided.ai/easyocr/)
- French to English Translation: [Helsinki-NLP/opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
- Speech synthesis (Text-to-Speech): [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)

### 4. Build Docker images
Next, build Docker images for each of the three microservices (OCR, Translation, TTS):
```bash
bash ./build_docker.sh
```
The `build_docker.sh` contains just 3 commands to automate the build of the 3 dockerfiles in the microservice folders. Each service (OCR, Translation, TTS) will be containerized and prepared to run as a microservice.

### 5. Run the Docker containers
Once the images are built, run the containers. This script will start the OCR, Translation, and TTS services, exposing them on different ports (OCR on port 8000, Translation on 8001, and TTS on 8002):

```bash
bash ./run_docker.sh
```
Check the status of the Docker containers to ensure that all three services are up and running:
```bash
docker ps
```
### 6. Run inferencing.ipynb to test the APIs
If you prefer working in VSCode, open the project in VSCode to access and modify the code or run the notebook:

```bash
code .
```
Now that the services are running, you can test the inferencing pipeline using the Jupyter notebook `inferencing.ipynb`. 

The notebook provides step-by-step code to:

1. Send an image to the OCR service.
2. Translate the extracted text using the translation service.
3. Generate speech using the TTS service.

Make sure you have a png with text on it saved in the working directory called `sample_image.png` in order to do the inferencing.