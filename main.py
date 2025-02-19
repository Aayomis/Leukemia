from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import gdown
import os

app = FastAPI()



# ✅ Correct Google Drive link format (change 'file/d/ID/view' to 'uc?id=ID')
MODEL_URL = "https://drive.google.com/uc?id=1C8FdGW2qV4A6sej8b9CNTkeYkfciNTUf"
MODEL_PATH = "leukemia_new.h5"

# Function to verify if the file exists and has a valid size
def is_valid_file(filepath, min_size_kb=100):  # 100 KB threshold to check if download was successful
    return os.path.exists(filepath) and os.path.getsize(filepath) > (min_size_kb * 1024)

# ✅ Check if model is already downloaded
if not is_valid_file(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Recheck file integrity before loading
if is_valid_file(MODEL_PATH):
    print("Loading model...")
    model_prod = tf.keras.models.load_model(MODEL_PATH)
else:
    raise RuntimeError("Model file is missing or corrupted. Download failed.")

CLASSNAMES = ['healthy blood smear', 'leukemia blood smear']

@app.get("/ping")
async def ping():
    return "Hello i am alive"

def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))

   return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0) 

    prediction = model_prod.predict(img_batch)

    predicted_class = CLASSNAMES[np.argmax(prediction[0])]

    confidence = np.max(prediction[0])

    return {
       'class':predicted_class,
       'confidence': float(confidence)}

   

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8000)