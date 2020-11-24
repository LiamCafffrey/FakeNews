import streamlit as st
import numpy as np
import pandas as pd
import trafilatura


st.title('Fake or Real?')
input_method = st.radio('Choose your input', ('Text', 'Link'))
if input_method == 'Text':
    title = st.text_input('Article title')
    text = st.text_area('Article body')
elif input_method == 'Link':
    url = st.text_input('Article URL')
st.button('Analyze')


def get_title_text_web(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    html = request.urlopen(url).read().decode('utf8')
    html[:60]
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title').string
    dictio={'title':[title], 'text':[text]}
    df =pd.DataFrame(dictio, columns=['title','text'])
    return df
