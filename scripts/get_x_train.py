import os
import pickle
import streamlit as st

def get_var():
    root_path = os.path.dirname(os.path.dirname(__file__))
    path_var = os.path.join(root_path,'raw_data','x_train.pkl')
    return pickle.load(open(path_var,"rb"))
