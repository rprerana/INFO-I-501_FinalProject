# INFO-I-501_FinalProject
This repository contains the files for the final project of my INFO-I501 course.

**Streamlit App** - https://info-i-501finalproject.streamlit.app/

The model used in the app is hosted on hugging face as it was too big to store on github.

**Link to model** - https://huggingface.co/preranar/my_awesome_model

**Abstract**

For my final project I have built a model that analyses a review and gives the polarity and sentiment of the review.

The most common application of sentiment analysis in NLP is the extraction and observation of opinions from a person or a group of people based on their own words, viewpoints, or observations of certain situations. This approach can be used to accurately forecast the text's emotions, like if it is positive or negative.

Coming to movie reviews, a textual movie review tells us about the strong and weak points of the movie and if the movie in general meets the expectations of the reviewer. Sentimental analysis can be used to determine the attitude of the reviewer with respect to the overall polarity of review.

Because the success or the failure of a movie depends on its reviews, there is a need to build a good sentiment analysis model that can classify movie reviews.

This project aims to develop a sentiment analysis model to evaluate the emotion and sentiment of a textual movie review.

**Data Description**

The dataset I have used to train the model is the IMDB 50K Dataset

Size of the Dataset - 66.21 MB

Link to the Dataset - [imdb-dataset-of-50k-movie-reviews](https://huggingface.co/datasets/imdb)

The dataset has two columns -

text: a string feature.

label: a classification label, with possible values including neg (0), pos (1).

**Algorithm Description**

The model was trained using a pre-trained model - DistilBERT base model (uncased). The model does not make a difference between english and English.

The following hyperparameters were used during training:

learning_rate: 2e-05

train_batch_size: 16

eval_batch_size: 16

seed: 42

optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08

lr_scheduler_type: linear

num_epochs: 2

**Tools Used**

I have used the following tools in the project

Github

Streamlit

Hugging Face

Google Colab

T4 GPU (through Colab)

I have used the following NLP python libraries:

torch

transformers

