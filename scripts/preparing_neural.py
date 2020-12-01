from nltk.stem import WordNetLemmatizer
from generic_func import df_apply
import pandas as pd
from keras.preprocessing.text import Tokenizer
from cleaning import apply_cleaning
from train_neural import split_train_test_data, fit_model, save_model
from train_neural import get_preprocessor
import pickle
import tensorflow as tf
import json,os
import keras

root_path = os.path.dirname(os.path.dirname(__file__))
tokenizer_path = os.path.join(root_path,'save_model','neural_tokenizer.json')

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

def embedding(df):
    with open(tokenizer_path) as f:
        tokenizer_json = f.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    df = tokenizer.texts_to_sequences(df.title_text)
    df = tf.keras.preprocessing.sequence.pad_sequences(df, padding='post', maxlen=256)
    return df
