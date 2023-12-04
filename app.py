import numpy as np
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

st.subheader('Enter text to be analysed')
text = st.text_area(label="")
a=''
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
lab = a[0]['label']
per = a[0]['score']
opp_per = 1-a[0]['score']
col1, col2 = st.columns(2)
with col1:
   st.header("Sentiment")
   st.subheader(lab)
with col2:
   st.header("%")
   st.subheader(per*100)

if(lab=='POSITIVE'):
   st.write('entered if ')
   labels = ["Positive", "Negative"]
   sizes = [per,opp_per]
elif(lab=='NEGATIVE'):
   st.write('entered else ')
   labels = 'Positive', 'Negative'
   sizes = [opp_per,per]

colors = ["#77DD76","#FF6962"]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
