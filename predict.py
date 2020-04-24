from fastapi import FastAPI, File, UploadFile
from starlette.responses import UJSONResponse
import zipfile
import os
import uvicorn
import tensorflow as tf
import numpy as np
import urllib
from PIL import Image
from io import BytesIO
import json

app = FastAPI()
prefix = os.environ.get('SCRIPT_PREFIX')

zipfile.ZipFile('fine_tuned_model.zip').extractall()
model = tf.saved_model.load('fine_tuned_model/saved_model')
model = model.signatures['serving_default']

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)


@app.post(prefix + "/file/", response_class=UJSONResponse)
async def classify(file: UploadFile = File(...)):
    img = await file.read()
    image = Image.open(BytesIO(img)).resize((600, 600))

    image = np.array(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    predictions = model(input_tensor)

    scores = predictions['detection_scores'].numpy()[0].tolist()
    classes = predictions['detection_classes'].numpy()[0].tolist()
    boxes = predictions['detection_boxes'].numpy()[0].tolist()

    results = []
    for score, label, box in zip(scores, classes, boxes):
        results.append(
            {
                "label": label_mapping[str(label)],
                "score": round(score, 10),
                "y_min": round(box[0], 10),
                "x_min": round(box[1], 10),
                "y_max": round(box[2], 10),
                "x_max": round(box[3], 10),
            }
        )

    return {'predictions': results}


@app.post(prefix + "/url/", response_class=UJSONResponse)
async def classify(data: dict):
    img = urllib.request.urlopen(data['url'])
    image = np.array(Image.open(img).resize((600, 600)))

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    predictions = model(input_tensor)

    scores = predictions['detection_scores'].numpy()[0].tolist()
    classes = predictions['detection_classes'].numpy()[0].tolist()
    boxes = predictions['detection_boxes'].numpy()[0].tolist()

    results = []
    for score, label, box in zip(scores, classes, boxes):
        label = int(float(label)) - 1
        results.append(
            {
                "label": label_mapping[label],
                "score": round(score, 10),
                "y_min": round(box[0], 10),
                "x_min": round(box[1], 10),
                "y_max": round(box[2], 10),
                "x_max": round(box[3], 10),
            }
        )

    return {'predictions': results}


@app.get(prefix + "/", response_class=UJSONResponse)
async def hello_world():
    """Hello world to check endpoint health"""
    return {"message": "hello world"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") # workers=4 ??
