from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np


app = FastAPI()

model = tf.keras.models.load_model("recipe-classification.h5")
class_names = ['burgers',
               'chicken biriyani',
               'chicken pakoda',
               'chicken roll',
               'chicken tandori',
               'chiken curry',
               'chole bhature ',
               'dal fry',
               'dosa',
               'egg curry',
               'fish curry',
               'noodles',
               'paneer curry',
               'pizza',
               'prawn curry',
               'rice',
               'samosa',
               'vada pav']


@app.get("/ping")
async def ping():
    return class_names[0]


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    resized_images = tf.image.resize(image, (256, 256))
    return resized_images


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = model.predict(img_batch)
    label = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'label': label,
        'confidence': int(float(confidence)*100)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
