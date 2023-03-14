import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# $WIPE_BEGIN

# $WIPE_END

from fastapi import FastAPI, File, UploadFile
from typing import List
from starlette.responses import Response
from tensorflow import convert_to_tensor
from tensorflow.keras import models
from PIL import Image

# Internal imports
from ml_logic import metrics

app = FastAPI()

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

# # Uploading images (vv, vh, wb)
@app.post("/upload")
async def receive_image(images: List[UploadFile]=File(...)):
    images_list = []
    ### Receiving and decoding the image
    for img in images:
        contents = await img.read()
        nparr = np.fromstring(contents, np.uint8)
        img_gs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_gs_norm = img_gs/255

        images_list.append(img_gs_norm) # extension depends on which format is sent from Streamlit

    for_test = convert_to_tensor(images_list)
    model_test = models.load_model('/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/model_for_api.h5',
                                 custom_objects={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})
    img_pred = model_test.predict(for_test)
    im_bytes = cv2.imencode('.png', img_pred[0])[1].tobytes()

    return Response(content=im_bytes, media_type="image/png")

# @app.post("/upload")
# async def receive_image(images: List[UploadFile]=File(...)):
#     images_list = []
#     ### Receiving and decoding the image
#     for img in images:
#         image = np.array(Image.open(img.file))
#         # contents = await img.read()
#         # nparr = np.fromstring(contents, np.uint8)
#         nparr_norm = image/255
#         img_gs = cv2.imdecode(nparr_norm, cv2.IMREAD_GRAYSCALE)
#         # img_gs_norm = img_gs/255

#         images_list.append(img_gs) # extension depends on which format is sent from Streamlit

#     for_test = convert_to_tensor(images_list)
#     model_test = models.load_model('/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/model_for_api.h5',
#                                  custom_objects={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})
#     img_pred = model_test.predict(for_test)
#     im_bytes = cv2.imencode('.png', img_pred[0])[1].tobytes()

#     return Response(content=im_bytes, media_type="image/png")

# @app.post("/upload")
# async def receive_image(images: List[UploadFile]=File(...)):
#     images_list = []
#     ### Receiving and decoding the image
#     for img in images:
#         contents = await img.read()
#         nparr = np.fromstring(contents, np.uint8)
#         img_gs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img_gs_norm = img_gs

#         images_list.append(img_gs_norm) # extension depends on which format is sent from Streamlit

#     # for_test = convert_to_tensor(images_list)
#     # model_test = models.load_model('/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/model_for_api.h5',
#     #                              custom_objects={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})
#     # img_pred = model_test.predict(for_test)
#     im_bytes = cv2.imencode('.png', images_list[1])[1].tobytes()

#     return Response(content=im_bytes, media_type="image/png")
