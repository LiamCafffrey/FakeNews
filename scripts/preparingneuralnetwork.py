import pandas as pd

def drop_columns(df):
    df.drop(columns = ['subject','date'], inplace = True)

def create_target(df, score):
    df['score'] = score

def concat(df1,df2):
    return pd.concat([df1,df2],ignore_index=True)

#data['title_text'] = data['title'] + " " + data['text']
#data.drop(columns = ['title', 'text'], inplace = True)

def df_concat_title_text(df):
    return df['title_text'] = df['title'] + " " + df['text']

def apply_preparing_merge(df):
    df_concat_title_text(df)
    return df.drop(columns = ['title', 'text'], inplace = True)

def apply_preparing(df_true, df_fake):
    drop_columns(df_true)
    drop_columns(df_fake)

    create_target(df_true,1)
    create_target(df_fake,0)

    new_data = concat(df_true, df_fake)
    apply_preparing_merge(new_data)

    return new_data


