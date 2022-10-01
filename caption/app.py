from sqlite3 import Time
import time
import streamlit as st
from preprocessor import *




st.markdown("<h1 style='text-align: center;'>Caption Recommendation System</h1>", unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("Upload a file")
    
if uploaded_file is not None:
    bytes_data = uploaded_file
    st.image(bytes_data)
    
    #st.button("Suggest Caption")
    if st.button("Suggest Caption"):
        st.title("Caption:"+ predict_new_image(uploaded_file))
        audio_file = open(audio(uploaded_file), 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        
  

