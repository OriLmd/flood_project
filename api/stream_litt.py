import streamlit as st
import requests

@st.cache(allow_output_mutation=True)
def get_session():
    return requests.Session()

@st.cache(allow_output_mutation=True)
def get_endpoint():
    return "http://localhost:8000/"

def app():
    st.title("My FastAPI App")
    session_state = get_session()
    endpoint = get_endpoint()
    st.write("Hello, world!")
    #text = st.text_input("Enter some text to classify:")
    #if st.button("Classify"):
    #    response = session_state.post(endpoint, json={"text": text})
    #    st.write(response.json())
