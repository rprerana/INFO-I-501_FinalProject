import numpy as np
import streamlit as st
from transformers import pipeline


st.write('Enter text to be analysed')
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
