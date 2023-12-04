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
dis = per*100
y = round(dis,2)
col1, col2 = st.columns(2)

html_str_p = f"""
<style>
p.a {{
  color:green;
  font:bold 40px Sans-serif;
}}
</style>
<p class="a">{lab}</p>
"""

html_str_n = f"""
<style>
p.b {{
  color:red;
  font:bold 40px Monospace;
}}
</style>
<p class="b">{lab}</p>
"""

with col1:
  st.header("SENTIMENT")
  if(lab=='POSITIVE'):
    st.markdown(html_str_p, unsafe_allow_html=True)
  else:
    st.markdown(html_str_n, unsafe_allow_html=True)
   
with col2:
   st.header("%")
   st.subheader(y)

if(lab=='POSITIVE'):
   labels = ["Positive", "Negative"]
   sizes = [per,opp_per]
elif(lab=='NEGATIVE'):
   labels = 'Positive', 'Negative'
   sizes = [opp_per,per]

colors = ["#77DD76","#FF6962"]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
