import pandas as pd

def convert(title,text):
    dictio = {'title':[title], 'text':[text]}
    df = pd.DataFrame(dictio, columns=['title','text'])
