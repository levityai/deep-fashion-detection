from fastapi import FastAPI
from starlette.responses import UJSONResponse
import zipfile
import os
import uvicorn
import tensorflow as tf
import numpy as np

app = FastAPI()
prefix = os.environ.get('SCRIPT_PREFIX')

# TODO: specify input (YAML) and unzip
model = tf.saved_model.load('saved_model')
model = model.signatures['serving_default']


@app.post(prefix + "/", response_class=UJSONResponse)
async def classify(payload: dict):
    data = payload["data"]

    img = urllib.urlopen(data['img_url'])
    image = np.array(Image.open(img))

    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    predictions = model(input_tensor)

    return {'predictions': predictions}

    # sample = {
    #     "boxes": [
    #         [
    #             {
    #                 "class": "worm1",
    #                 "score": "0.98123425",
    #                 "x_min": "234",
    #                 "x_max": "545",
    #                 "y_min": "13",
    #                 "y_max": "178",
    #             },
    #             {
    #                 "class": "worm2",
    #                 "score": "0.789345",
    #                 "x_min": "234",
    #                 "x_max": "545",
    #                 "y_min": "13",
    #                 "y_max": "178",
    #             }
    #         ],
    #         [
    #             {
    #                 "class": "worm2",
    #                 "score": "0.98123425",
    #                 "x_min": "234",
    #                 "x_max": "545",
    #                 "y_min": "13",
    #                 "y_max": "178",
    #             },
    #             {
    #                 "class": "worm3",
    #                 "score": "0.01453564",
    #                 "x_min": "234",
    #                 "x_max": "545",
    #                 "y_min": "13",
    #                 "y_max": "178",
    #             }
    #         ]
    #     ]
    # }


@app.get(prefix + "/", response_class=UJSONResponse)
async def hello_world():
    """Hello world to check endpoint health"""
    print(model.predict('bad book')[1].tolist())
    return {"message": "hello world"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
