import numpy as np
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt


text = st.text_area('Enter text to be analysed')
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
lab = a[0]['label']
per = a[0]['score']
opp_per = 1-a[0]['score']
st.write(lab)
if(lab=='POSITIVE'):
  st.write('entered if ')
  labels = ["Negative", "Positive"]
  sizes = [opp_per,per]
elif(lab=='NEGATIVE'):
  st.write('entered else ')
  labels = 'Positive', 'Negative'
  sizes = [opp_per,per]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
