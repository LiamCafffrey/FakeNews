from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle,os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

path = os.path.join('..','raw_data','pipeline_neural')

def get_x(df):
    return df['title_text']

def get_y(df):
    return df['score']

def split_train_test_data(df):
    x_train, x_test, y_train, y_test = train_test_split(get_x(df), get_y(df),random_state=0,test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_preprocessor(x1,x2):

    max_vocab = 25000
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x1)
    x_train = tokenizer.texts_to_sequences(x1)
    x_test = tokenizer.texts_to_sequences(x2)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=256)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=256)

    return x_train,x_test



def fit_model(x_train, y_train):

    max_vocab = 25000
    model = Sequential()
    model.add(layers.Embedding(max_vocab, output_dim=128))  # We have 25000 words in the vocabulary,
    model.add(layers.Masking())                                                     # and each word is represnted by a vector of size 128
    model.add(layers.LSTM(128, activation='tanh'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.3, callbacks=[es])

    return model


def save_model(pipeline):
 #   with open(path, "wb") as file:
 #           pickle.dump(pipeline, file)
    pipeline.save(path)
