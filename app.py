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
st.write(opp_per)
st.write(per)
if(lab=='POSITIVE'):
  st.write('entered if ')
  labels = 'Positive', 'Negative'
  sizes = [per,opp_per]
  #explode = (0.8, 0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')
elif(lab=='NEGATIVE'):
  labels = 'Positive', 'Negative'
  sizes = [per,opp_per]
  explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
