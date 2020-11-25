import os
from tensorflow import keras


def load_predict_neural():
    path = os.path.join('..','raw_data','pipeline_neural')
    model = keras.models.load_model(path)
    return model
