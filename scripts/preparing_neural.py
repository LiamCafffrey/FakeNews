from nltk.stem import WordNetLemmatizer
from generic_func import df_apply
import pandas as pd

def df_concat_title_text(df):
    df['title_text'] = df['title_clean'] + " " + df['text_clean']
    return df[['title_text', 'score']]

def apply_preparing_merge(df):
    df_concat_title_text(df)
    return df[['title_text','score']]

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmas= []
    for word in text.split():
        lemmas.append(lemmatizer.lemmatize(word))
    return " ".join(lemmas)

def apply_lemmatize(df):
    df_apply(df,'title_text', 'title_text', lemmatize)
    return df[['title_text', 'score']]
