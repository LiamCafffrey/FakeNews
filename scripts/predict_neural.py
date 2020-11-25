import os

def load_predict():
    path = os.path.join('..','raw_data','pipeline_neural.pkl')
    my_pipeline_neural = pickle.load(open(path,"rb"))
