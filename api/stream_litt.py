import streamlit as st
import requests
import json
import numpy as np
from io import BytesIO

st.title("We can make a flood mask")

option = st.selectbox('What operation',
                      ('flood','not flood'))

#st.write("")
st.write("How much flood do you want?")
x = st.slider("Flood", 0,100,30)

inputs = {"flood":option , "x":x}

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

    #x = np.arange(28*28).reshape(28, 28)

    # save in to BytesIo buffer
    np_bytes = BytesIO()
    np.save(np_bytes, bytes_data, allow_pickle=True)

    # get bytes value
    np_bytes = np_bytes.getvalue()

    # load from bytes into numpy array
    load_bytes = BytesIO(np_bytes)
    loaded_np = np.load(load_bytes, allow_pickle=True)

    st.write("filename:", uploaded_file.name)
    st.write(np_bytes)
    st.write(loaded_np)



if st.button('Generate Flood'):
    res = requests.post(url = "http://127.0.0.1:8000/test", data = json.dumps(inputs))
st.subheader(f"Does our model make a flood? = {res.text}")
