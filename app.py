import numpy as np
import streamlit as st
from transformers import pipeline


text = st.text_area('Enter text to be analysed')
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
lab = a[0]['label']
per = a[0]['score']
opp_per = 1-a[0]['score']
st.write(opp_per)
st.write(per)
