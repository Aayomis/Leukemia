from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import gdown
import os

app = FastAPI()


MODEL_URL = "https://drive.google.com/file/d/1C8FdGW2qV4A6sej8b9CNTkeYkfciNTUf"
MODEL_PATH = "leukemia_new.h5"

# Download the model if not already downloaded
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, use_cookies=False)

# Load the model
model_prod = tf.keras.models.load_model(MODEL_PATH)

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