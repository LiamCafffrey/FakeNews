import re, string
from nltk.corpus import stopwords
from generic_func import df_apply

def drop_na(df):
    df.dropna(inplace = True)
    return df

def pattern_getty():
    stop_words = ['/Getty Images', 'Reuters', 'reuters']
    return '|'.join(r"\b{}\b".format(x) for x in stop_words)

def clean_getty_image(df, column):
    df[column] = df[column].str.replace(pattern_getty(), '')
    return df

def rem_urls(text):
    return re.sub('https?:\S+','',text)

punc_no_sq = '!“#$%&\()*+,./:;<=>?@[\\]^_`{|}~“”—’-"'
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

def stopwords_ratio(tokens):
    count_stop_words = 0
    amount_tokens = 0
    for token in tokens:
        amount_tokens += 1
        if token in stop_words:
            count_stop_words += 1
    if amount_tokens == 0:
        return 0
    return count_stop_words / amount_tokens

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in (stop_words)])

def tokenize_text(text):
    return text.split()

def title_length_char(df):
    df['title_length_char'] = df.title.str.len()
    return df

def title_upper(df):
    df['title_Upper'] = df['title'].str.count(r'[A-Z]')
    return df

def title_Upper_Ratio(df):
    df['title_Upper_Ratio'] = df['title_Upper']/df['title_length_char']
    return df

def text_tokens_with_stopwords(df):#ATTENZIONE TYPO!!!
    df_apply(df,'text', 'text_tokens_with_stopwords', tokenize_text)
    return df

def text_stop_words_ratio(df):
    df_apply(df,'text_tokens_with_stopwords', 'text_stop_words_ratio', stopwords_ratio)
    return df

def apply_cleaning(df):
    drop_na(df)
    clean_getty_image(df,'text')
    df_apply(df,'title', 'title_clean', rem_urls)
    df_apply(df,'text', 'text_clean', rem_urls)
    df_apply(df, 'text_clean', 'text_clean', remove_punctuation)
    df_apply(df, 'title_clean', 'title_clean', remove_punctuation)
    df_apply(df,'title_clean', 'title_clean', remove_numbers)
    df_apply(df,'text_clean', 'text_clean', remove_numbers)

    title_length_char(df)
    title_upper(df)
    title_Upper_Ratio(df)
    text_tokens_with_stopwords(df)
    text_stop_words_ratio(df)

    df_apply(df,'title_clean','title_clean', lower_case)
    df_apply(df,'text_clean', 'text_clean', lower_case)

    df_apply(df,'title_clean','title_clean', remove_stopwords)
    df_apply(df,'text_clean','text_clean', remove_stopwords)
    df_apply(df,'title_clean','title_tokens', tokenize_text)
    df_apply(df,'text_clean','text_tokens', tokenize_text)

    return df
