import streamlit as st
from PIL import Image
import requests
#from dotenv import load_dotenv
import os
import numpy as np

# Set page tab display
st.set_page_config(
   page_title="Flood Detection using U-Net",
   page_icon= '🌊',
   layout="wide",
   initial_sidebar_state="expanded",
)

# Example local Docker container URL
# url = 'http://api:8000'
# Example localhost development URL
url = 'http://localhost:8000'
#load_dotenv()
#url = os.getenv('API_URL')


# App title and description
st.header('🌊 Flood Detection using U-Net 🌊')


### Create a native Streamlit file upload input
st.subheader("Upload your images and see what the results are with a flood👇")

#upload vv,vh and wb images
cola, colb, colc = st.columns(3)
with cola:
    img_vv = st.file_uploader('Upload vv image', type = 'png', accept_multiple_files=False)
with colb:
    img_vh = st.file_uploader('Upload vh image', type = 'png', accept_multiple_files=False)
    st.write('')
with colc:
    img_wb = st.file_uploader('Upload wb image', type = 'png', accept_multiple_files=False)


#once images are uploaded send for prediction

if st.button('Get Prediction'):
    img_file_buffer = [img_vv,img_vh,img_wb]
    image_data = []
    for image in img_file_buffer:
        image_data.append(('files',image))
    res = requests.post(url + "/upload", files = image_data)

    #this is for centering the image
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(res.content)

    with col3:
        st.write(' ')
    #adding text below image to describe the image
    st.subheader("Here is what this region looks like if it were flooded ☝️")
else:
    #while images aren't uploaded it will display this messages
    st.write('🚨 Click this button only when images are uploaded 🚨')
