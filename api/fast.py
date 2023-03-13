import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# $WIPE_BEGIN

#from taxifare.ml_logic.registry import load_model
#from taxifare.ml_logic.preprocessor import preprocess_features
# $WIPE_END

from fastapi import FastAPI, File, UploadFile
from typing import List
from starlette.responses import Response
from tensorflow import convert_to_tensor
from tensorflow.keras import models
from PIL import Image

# Internal imports
from ml_logic import metrics

# from fastapi.staticfiles import StaticFiles

app = FastAPI()

## Optional, good practice for dev purposes. Allow all middlewares
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # Allows all origins
#    allow_credentials=True,
#    allow_methods=["*"],  # Allows all methods
#    allow_headers=["*"],  # Allows all headers
#)

# $WIPE_BEGIN
#💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!
# This will prove very useful for demo days
#app.state.model = load_model()
#$WIPE_END

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
#@app.get("/predict")
#def predict(pickup_datetime: str,  # 2013-07-06 17:18:00
#            pickup_longitude: float,    # -73.950655
#            pickup_latitude: float,     # 40.783282
#            dropoff_longitude: float,   # -73.984365
#            dropoff_latitude: float,    # 40.769802
#            passenger_count: int):      # 1
#    """
#    Make a single course prediction.
#    Assumes `pickup_datetime` is provided as string by the user in "%Y-%m-%d %H:%M:%S" format
#    Assumes `pickup_datetime` implicitely refers to "US/Eastern" timezone (as any user in New York City would naturally write)
#    """
#    # $CHA_BEGIN
#
#    # Convert to US/Eastern TZ-aware!
#    pickup_datetime_localized = pd.Timestamp(pickup_datetime, tz='US/Eastern')
#
#    X_pred = pd.DataFrame(dict(
#        pickup_datetime=[pickup_datetime_localized],
#        pickup_longitude=[pickup_longitude],
#        pickup_latitude=[pickup_latitude],
#        dropoff_longitude=[dropoff_longitude],
#        dropoff_latitude=[dropoff_latitude],
#        passenger_count=[passenger_count]))
#
#    model = app.state.model
#    assert model is not None
#
#    X_processed = preprocess_features(X_pred)
#    y_pred = model.predict(X_processed)
#
#    # ⚠️ fastapi only accepts simple python data types as a return value
#    # among which dict, list, str, int, float, bool
#    # in order to be able to convert the api response to json
#    return dict(fare_amount=float(y_pred))
#    # $CHA_END
#

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

# Good one
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
    model_test = models.load_model('/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/first_unet_2000_20230310-143412.h5',
                                 custom_objects={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})
    img_pred = model_test.predict(for_test)
    im_bytes = cv2.imencode('.png', img_pred[0])[1].tobytes()

    return Response(content=im_bytes, media_type="image/png")

# @app.post("/upload")
# async def receive_image(images: List[UploadFile]=File(...)):

#     images_list = []
#     ### Receiving and decoding the image
#     for img in images:
#         contents = await img.read()
#         nparr = np.fromstring(contents, np.uint8)
#         cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

#         ### Encoding and responding with the image
#         images_list.append(cv2.imencode('.png', cv2_img)[1]) # extension depends on which format is sent from Streamlit

#     return Response(content=images_list[1].tobytes(), media_type="image/png")

# @app.post("/upload")
# async def receive_image(images: List[UploadFile]=File(...)):
#     images_list = []
#     ### Receiving and decoding the image
#     for img in images:
#         contents = await img.read()
#         nparr = np.fromstring(contents, np.uint8)
#         img_gs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img_gs_norm = img_gs/255

#         images_list.append(img_gs_norm) # extension depends on which format is sent from Streamlit

#     for_test = convert_to_tensor(images_list)
#     model_test = models.load_model('/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/first_unet_2OOO_v2_20230310-171824.h5',
#                                  custom_objects={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})
#     img_pred = model_test.predict(for_test)
#     # im = cv2.imencode('.png', img_pred)[1]

#     return Response(content=img_pred[1].tobytes(), media_type="image/png")
#     # return {'type':str(type(img_pred)),'shape':img_pred.shape}

# @app.post("/upload")
# async def receive_image(images: List[UploadFile]=File(...)):
#     images_list = []
#     ### Receiving and decoding the image
#     for img in images:
#         contents = await img.read()

#         nparr = np.fromstring(contents, np.uint8)
#         cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img_gs_norm = cv2_img/255
#         #nparr = np.fromstring(contents, np.uint8)
#         #cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
#         ### Encoding and responding with the image
#         images_list.append(img_gs_norm) # extension depends on which format is sent from Streamlit

#     for_test = convert_to_tensor(images_list)
#     model_test = models.load_model("/Users/mariottecharles-etienne/code/OriLmd/flood_project/models/first_unet_2OOO_v2_20230310-171824.h5",
#                                    custom_objects = {"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()})

#     img_pred = model_test.predict(for_test)
#     #im = Image.fromarray(img_pred)
#     #im.save('pred.png')
#     #nparr_pred = np.fromstring(img_pred, np.uint8)
#     #cv2_img_pred = cv2.imread(nparr_pred, cv2.IMREAD_COLOR)
#     return Response(content=img_pred[1,:,:,0].tobytes(), media_type="image/png")
#     #return Response(content=im.tobytes(), media_type="image/png")
