import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from typing import List
from starlette.responses import Response
from tensorflow import convert_to_tensor, concat, io, data
from tensorflow.keras import models
from PIL import Image
#from ml_logic.load_preprocess import read_one_image
from ml_logic import metrics
from ml_logic.results import result_into_class


app = FastAPI()

@app.get("/")
def root():
    return dict(greeting="Hello")
    

@app.post("/upload")
async def receive_image(files: List[UploadFile]=File(...)):

    images_list = []

    # Receiving and decoding the image
    for img in files:
        contents = await img.read()
        #contents = io.read_file(img)
        img_one_channel = io.decode_png(contents, channels=1) # channels=1 to have a grayscale image
        img_norm = img_one_channel/255

        images_list.append(img_norm)


    img_for_test = concat(images_list,axis=-1)

    # Load the model
    model_test = models.load_model('model_for_api.h5',
                                 custom_objects={"DiceLoss": metrics.DiceLoss(),"Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})

    # Predict
    img_pred = model_test.predict(np.expand_dims(img_for_test,axis=0))


    #Choose according to threshold
    img_pred_threshold = result_into_class(img_pred)*255

    # Transform predicted image to bytes in order to display it
    im_bytes = cv2.imencode('.png', img_pred_threshold[0])[1].tobytes()

    return Response(content=im_bytes, media_type="image/png")
