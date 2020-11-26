from nltk.stem import WordNetLemmatizer
from scripts.generic_func import df_apply
import pandas as pd
from keras.preprocessing.text import Tokenizer
from scripts.cleaning import apply_cleaning
from scripts.train_neural import split_train_test_data, fit_model, save_model
from scripts.train_neural import get_preprocessor
import pickle
from scripts.get_x_train import get_var
import tensorflow as tf



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
    x_train = get_var()
    max_vocab = 25000
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x_train)
    df = tokenizer.texts_to_sequences(df.title_text)
    df = tf.keras.preprocessing.sequence.pad_sequences(df, padding='post', maxlen=256)
    return df
