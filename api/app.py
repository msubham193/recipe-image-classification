# from fastapi import FastAPI, File, UploadFile
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit


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


@app.route("/")
def ping():
    return "Connection established !"


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    resized_images = tf.image.resize(image, (256, 256))
    return resized_images


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    data = file.read()
    image = read_file_as_image(data)
    img_batch = np.expand_dims(image, 0)
    prediction = model.predict(img_batch)
    label = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'label': label,
        'confidence': int(float(confidence)*100)
    }


if __name__ == "__main__":
    app.run()
