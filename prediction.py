import io

import numpy as np
import pybase64
import tensorflow as tf
import uvicorn
from PIL import Image
from fastapi import FastAPI, File
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from pydantic import BaseModel
from starlette.requests import Request

app = FastAPI()
path = "F:/Inzynierka/Images-preprocessing/Lung-model.h5"


class ProbabilityResponse(BaseModel):
    probabilityHealthy: float


class LoadModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict_data(self, img):
        preds = self.model.predict(img)
        return preds


def prepare_image(image, target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def load_model(path):
    return tf.keras.models.load_model(path)


model = LoadModel(path)


@app.post("/predict")
async def predict(request: Request,
                  file: str = File(...)):
    data = {"success": False}

    if request.method == "POST":

        image = Image.open(io.BytesIO(pybase64.b64decode(file)))
        image = prepare_image(image, target=(224, 224))
        preds = model.predict_data(image)
        data = {"probabilityHealthy": float(preds[0])}

    return data
if __name__ == '__main__':

    uvicorn.run(app, port=8000)
