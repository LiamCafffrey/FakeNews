import os

def load_predict():
    path = os.path.join('..','raw_data','pipeline_logistic.pkl')
    my_pipeline_logistic = pickle.load(open(path,"rb"))
