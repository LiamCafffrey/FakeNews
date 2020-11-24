import pandas as pd

def drop_columns(df):
	df.drop(columns = ['subject','date'], inplace = True)

def create_target(df, score):
	df['score'] = score

def concat(df1,df2):
	return pd.concat([df1,df2],ignore_index=True) 

def apply_preparing(df_true, df_fake):
	drop_columns(df_true)
	drop_columns(df_fake)

	create_target(df_true,1)
	create_target(df_fake,0)

	return concat(df_true, df_fake)

