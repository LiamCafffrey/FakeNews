import os
import pickle
import streamlit as st

def load_predict_logistic():
    root_path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root_path,'raw_data','logistic_model.pkl')
    return pickle.load(open(path,"rb"))

