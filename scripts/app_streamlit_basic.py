import streamlit as st
import numpy as np
import pandas as pd
import scripts
from predict_logistic import load_predict_logistic
from predict_neural import load_predict_neural
from text_extractor import get_title_text_web
from cleaning import apply_cleaning
from generic_func import df_apply
from typo_func import apply_typo_ratio
from tensorflow import keras
from convert_input_to_df import convert
from preparing_neural import apply_lemmatize
from preparing_neural import apply_preparing_merge
from preparing_neural import lemmatize
from preparing_neural import embedding



neural_model = load_predict_neural()
logistic_model = load_predict_logistic()


st.title('Fake or Real?')
input_method = st.radio('Choose your input', ('Text', 'Link'))
if input_method == 'Text':
    title = st.text_input('Article title')
    text = st.text_area('Article body')
elif input_method == 'Link':
    url = st.text_input('Article URL')


analyze_status_logistic = st.button('Analyze_with_Logistic')
analyze_status_neural = st.button('Analyze_with_Neural')


###### Logistic
if input_method == 'Text' and analyze_status_logistic == True:

    input_df = convert(title,text)
    input_df = apply_cleaning(input_df)

    input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio','text_stop_words_ratio']]
    prediction = logistic_model.predict(input_df)
    if prediction == 1:
        st.write('I think its true')
    else:
        st.write('I think its fake')

if input_method == 'Link' and analyze_status_logistic == True:
    input_df = get_title_text_web(url)
    input_df = apply_cleaning(input_df)


    input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio','text_stop_words_ratio']]



    prediction = logistic_model.predict(input_df)
    if prediction == 1:
        st.write('I think its true')
    else:
        st.write('I think its fake')


###### Neural network
if input_method == 'Text' and analyze_status_neural == True:

    input_df = convert(title,text)
    input_df = apply_cleaning(input_df)
    input_df['title_text'] = input_df['title_clean'] + " " + input_df['text_clean']
    input_df = input_df[['title_text']]
    input_df['title_text'] = input_df['title_text'].apply(lemmatize)
    input_embedded = embedding(input_df)
    prediction = neural_model.predict(input_embedded)
    if prediction >= 0.5:
        st.write('I think its true')
    else:
        st.write('I think its fake')

if input_method == 'Link' and analyze_status_neural == True:

    input_df = get_title_text_web(url)
    input_df = apply_cleaning(input_df)
    input_df['title_text'] = input_df['title_clean'] + " " + input_df['text_clean']
    input_df = input_df[['title_text']]
    input_df['title_text'] = input_df['title_text'].apply(lemmatize)
    input_embedded = embedding(input_df)
    prediction = neural_model.predict(input_embedded)

    if prediction >= 0.5:
       st.write('I think its true')
    else:
       st.write('I think its fake')



