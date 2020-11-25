import trafilatura
from bs4 import BeautifulSoup
from urllib import request
import pandas as pd

def get_title_text_web(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    html = request.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title').string
    dictio = {'title':[title], 'text':[text]}
    df = pd.DataFrame(dictio, columns=['title','text'])
    return df
