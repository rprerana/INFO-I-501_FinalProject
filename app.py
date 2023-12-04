import numpy as np
import streamlit as st
from transformers import pipeline


st.write('Enter text to be analysed')
text = "After a good, but not great start in the franchise, McQuarrie creates the best combination of the good that came before him, with a better script that even MI-I, the unstoppable action of MI-3, good humor without going over the top contrary to MI-4, and those romantic touches that help the movie to slow down and feel a bit for the characters. I think there is one part of the movie, Vanessa Kirby's character and its plot, that is really questionable. Other than that, the movie is just a great all-around action blockbuster."
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
