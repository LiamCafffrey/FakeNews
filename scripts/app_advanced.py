import streamlit as st
import pandas as pd
import numpy as np


import scripts
from predict_logistic import load_predict_logistic
from predict_neural import load_predict_neural
from text_extractor import get_title_text_web
from cleaning import apply_cleaning
from generic_func import df_apply
from typo_func import apply_typo_ratio

from preparing_neural import apply_preparing_merge, apply_lemmatize, lemmatize

from load_css import local_css
from preparing_neural import embedding
from convert_input_to_df import convert

local_css("style.css")



def get_title_text_input(title, text):
	return pd.DataFrame({"title": [title], "text": [text]});

def show_man(col, fake):
	with col:
		if not fake:
			st.image('../illustrations/sir_real.png', use_column_width=True)
		else:
			st.image('../illustrations/sir_fake.png', use_column_width=True)



def show_lady(col, fake):
	with col:
		if not fake:
			st.image('../illustrations/lady_real.png', use_column_width=True)
		else:
			st.image('../illustrations/lady_fake.png', use_column_width=True)


logistic_model = load_predict_logistic()
neural_model = load_predict_neural() #OSError: SavedModel file does not exist at: ../raw_data/neural_model/{saved_model.pbtxt|saved_model.pb}

title = None
text = None
url = None


st.title('Real or Fake?')

input_method = st.radio('Choose your input', ('Text', 'Link'))

if input_method == 'Text':
	title = st.text_input('Article - News title')
	text = st.text_area('Article - News body')

elif input_method == 'Link':
	url = st.text_input('Article - News URL')


if st.button('Analyze'):
	col1, col2 = st.beta_columns(2)

	if input_method == 'Text':

		if len(title) <12 and len(text) < 101:
			st.image('../illustrations/new_input.png', use_column_width=True)

		if len(title) > 11 and len(text) > 101:
			input_df = convert(title,text)
			input_df = apply_cleaning(input_df)
			input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio','text_stop_words_ratio']]
			prediction = logistic_model.predict(input_df)

			show_man(col1, prediction[0] == 0)

			input_df = convert(title,text)
			input_df = apply_cleaning(input_df)
			input_df['title_text'] = input_df['title_clean'] + " " + input_df['text_clean']
			input_df = input_df[['title_text']]
			input_df['title_text'] = input_df['title_text'].apply(lemmatize)
			input_embedded = embedding(input_df)
			prediction = neural_model.predict(input_embedded)

			show_lady(col2,  prediction[0] <= 0.5 )


	elif input_method == 'Link':

		if len(url) < 20:
			st.image('../illustrations/new_input.png', use_column_width=True)

		if len(url) > 19:
			input_df = get_title_text_web(url)
			input_df = apply_cleaning(input_df)
			input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio','text_stop_words_ratio']]
			prediction = logistic_model.predict(input_df)

			show_man(col1, prediction[0] == 0)

			input_df = get_title_text_web(url)
			input_df = apply_cleaning(input_df)
			input_df['title_text'] = input_df['title_clean'] + " " + input_df['text_clean']
			input_df = input_df[['title_text']]
			input_df['title_text'] = input_df['title_text'].apply(lemmatize)
			input_embedded = embedding(input_df)
			prediction = neural_model.predict(input_embedded)


			show_lady(col2,  prediction[0] <= 0.5 )





else:
	st.image('../illustrations/start_image.png', use_column_width=True)



# st.components.v1.html("<img src='./illustrations/start_image.png'/>")
