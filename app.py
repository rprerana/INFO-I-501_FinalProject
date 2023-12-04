import numpy as np
import streamlit as st
from transformers import pipeline


st.write('Enter text to be analysed')
text = "After a good, but not great start in the franchise, McQuarrie creates the worst combination of the bad that came before him, with a mediocore script worse that MI-I, the below average action of MI-3, humor going over the top, and those unnecessary romantic touches that do not help the movie. I think there is one part of the movie, Vanessa Kirby's character and its plot, that is really questionable. Other than that, the movie is just an average action blockbuster."
classifier = pipeline("sentiment-analysis", model="preranar/my_awesome_model")
a = classifier(text)
st.write(a)
