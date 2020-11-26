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

from load_css import local_css

local_css("style.css")



# Title_html = """
#     <style>
#         body {
#         	background-color: beige;																	
#         }
#     </style> 
#     """
# st.markdown(Title_html, unsafe_allow_html=True) #Title rendering


# col1, col2 = st.beta_columns(2)

# with col1:
# 	st.button('Text')

# with col2:
# 	st.button('Link')

# if input_method == 'Link' and analyze_status_logistic == True:
#     input_df = get_title_text_web(url)
#     input_df = apply_cleaning(input_df)
#     input_df = apply_typo_ratio(input_df)
#     input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio', 'text_typo_ratio','text_stop_words_ratio']]
#     prediction = pipeline.predict(input_df)
#     if prediction == 1:
#         st.write('I think its true')
#     else:
#         st.write('I think its fake')

# def get_title_text_web(url):
#     downloaded = trafilatura.fetch_url(url)
#     text = trafilatura.extract(downloaded)
#     html = request.urlopen(url).read().decode('utf8')
#     # html[:60]
#     soup = BeautifulSoup(html, 'html.parser')
#     title = soup.find('title').string
#     # dictio = {'title':[title], 'text':[text]}
#     # df = pd.DataFrame(dictio, columns=['title','text'])
#     return title, text

def create_df(title, text): 
	df = pd.DataFrame({"title": [title], "text": [text]});

	apply_cleaning(df)

	return df


logistic_model = load_predict_logistic()


st.title('Real or Fake?')

input_method = st.radio('Choose your input', ('Text', 'Link'))

if input_method == 'Text':

	title = st.text_input('Article title')
	text = st.text_area('Article body')

	if len(title) > 0 and len(text) > 0:
		df = create_df(text, title)


elif input_method == 'Link':

	url = st.text_input('Article URL')
	if len(url) > 0:
		input_df = get_title_text_web(url)
	    # input_df = apply_cleaning(input_df)
	    # input_df = apply_typo_ratio(input_df)
	    # input_df = input_df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio', 'text_typo_ratio','text_stop_words_ratio']]
	    # input_df

if st.button('Analyze'):
	col1, col2 = st.beta_columns(2)
	with col1:
		st.image('./illustrations/sir_fake.png', use_column_width=True)

	with col2:
		st.image('./illustrations/lady_real.png', use_column_width=True)
else:
	st.image('./illustrations/start_image.png', use_column_width=True)



# st.components.v1.html("<img src='./illustrations/start_image.png'/>")

