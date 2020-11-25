import os
import pickle

def load_predict_logistic():
    root_path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root_path,'raw_data','pipeline_logistic.pkl')
    return pickle.load(open(path,"rb"))

