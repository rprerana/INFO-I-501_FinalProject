import numpy as np
import streamlit as st
from transformers import pipeline


text = st.text_area('Enter text to be analysed')
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
st.write(a{0}[0])
st.write(a{0}[1])
