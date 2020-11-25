import pandas as pd

def df_concat_title_text(df):
    return df['title_text'] = df['title'] + " " + df['text']

def apply_preparing_merge(df):
    df_concat_title_text(df)
    return df.drop(columns = ['title', 'text'], inplace = True)



