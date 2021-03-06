import re, string
from nltk.corpus import stopwords
from df_generic import df_apply


def pattern_getty():
	stop_words = ['/Getty Images']

	return '|'.join(r"\b{}\b".format(x) for x in stop_words)

def clean_getty_image(df, column):
	df[column] = df[column].str.replace(pattern_getty(), '') 	

def rem_urls(text):
    return re.sub('https?:\S+','',text)

punc_no_sq = '!“#$%&\()*+,./:;<=>?@[\\]^_`{|}~“”—’-'

def remove_punctuation(text):
    for punctuation in punc_no_sq:
        text = text.replace(punctuation, '')
    return text      

def remove_numbers(text):
    text = ''.join(word for word in text if not word.isdigit())
    return text

def lower_case(text):
    text = text.lower()
    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
	return ' '.join([word for word in text.split() if word.lower() not in (stop_words)])    

def tokenize_text(text):
    return text.split()   

def apply_cleaning(df):
	clean_getty_image(df,'text')

	df_apply(df,'title', 'title_clean', rem_urls)
	df_apply(df,'text', 'text_clean', rem_urls)

	df_apply(df,'title_clean', 'title_clean', remove_numbers)
	df_apply(df,'text_clean', 'text_clean', remove_numbers)

	df_apply(df,'title_clean','title_clean', lower_case)
	df_apply(df,'text_clean', 'text_clean', lower_case)
	
	df_apply(df,'title_clean','title_clean', remove_stopwords)
	df_apply(df,'text_clean','text_clean', remove_stopwords)

	df_apply(df,'title_clean','title_tokens', tokenize_text)
	df_apply(df,'text_clean','text_tokens', tokenize_text)


