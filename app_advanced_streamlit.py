import streamlit as st
import pandas as pd
import numpy as np


import scripts
from scripts.predict_logistic import load_predict_logistic
from scripts.predict_neural import load_predict_neural
from scripts.text_extractor import get_title_text_web
from scripts.cleaning import apply_cleaning
from scripts.generic_func import df_apply
from scripts.typo_func import apply_typo_ratio

from scripts.preparing_neural import apply_preparing_merge, apply_lemmatize

from load_css import local_css

local_css("style.css")



def get_title_text_input(title, text): 
	return pd.DataFrame({"title": [title], "text": [text]});

def show_man(col, fake):
	with col:
		if not fake:
			st.image('./illustrations/sir_real.png', use_column_width=True)
		else:	
			st.image('./illustrations/sir_fake.png', use_column_width=True)



def show_lady(col, fake):
	with col:
		if not fake:
			st.image('./illustrations/lady_real.png', use_column_width=True)
		else:	
			st.image('./illustrations/lady_fake.png', use_column_width=True)


logistic_model = load_predict_logistic()
# neural_model = load_predict_neural() #OSError: SavedModel file does not exist at: ../raw_data/neural_model/{saved_model.pbtxt|saved_model.pb}

title = None
text = None
url = None


st.title('Real or Fake?')

input_method = st.radio('Choose your input', ('Text', 'Link'))

if input_method == 'Text':
	title = st.text_input('Article title')
	text = st.text_area('Article body')
	
elif input_method == 'Link':
	url = st.text_input('Article URL')


if st.button('Analyze'):
	col1, col2 = st.beta_columns(2)

	if input_method == 'Text':

		if len(title) <12 and len(text) < 101:
			st.image('./illustrations/new_input.png', use_column_width=True)

		if len(title) > 11 and len(text) > 101:
			input_df = get_title_text_input(text, title)
			input_df = apply_cleaning(input_df)
			input_df = apply_typo_ratio(input_df)
			input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio', 'text_typo_ratio','text_stop_words_ratio']]
			prediction = logistic_model.predict(input_df)

			show_man(col1, prediction[0] == 0)

			input_df = apply_preparing_merge(text, title)
			input_df = apply_lemmatize(input_df)
			prediction = lneural_model.predict(input_df)
			
			show_lady(col2,  prediction[0] == 0 )


	elif input_method == 'Link':

		if len(url) < 20:
			st.image('./illustrations/new_input.png', use_column_width=True)

		if len(url) > 19:
			input_df = get_title_text_web(url)
			input_df = apply_cleaning(input_df)
			input_df = apply_typo_ratio(input_df)
			input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio', 'text_typo_ratio','text_stop_words_ratio']]
			prediction = logistic_model.predict(input_df)

			show_man(col1, prediction[0] == 0)

			input_df = apply_preparing_merge(text, title)
			input_df = apply_lemmatize(input_df)
			prediction = lneural_model.predict(input_df)
			
			show_lady(col2,  prediction[0] == 0 )



	
		
else:
	st.image('./illustrations/start_image.png', use_column_width=True)



# st.components.v1.html("<img src='./illustrations/start_image.png'/>")

