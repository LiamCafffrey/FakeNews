import enchant
from scripts.generic_func import df_apply

def create_get_typos_count():
	english = enchant.DictWithPWL("en_US", "vocab.txt")
	wrong_words={}
	correct_words=set()
	def get_typos_count(tokens):
	     wrong_count=0
	     for token in tokens:
	            if token in wrong_words:
	                wrong_words[token]+=1
	                wrong_count+=1
	            else:
	                if not token in correct_words:
	                    if token[0].islower() and not '-' in token and not english.check(token) and not english.check(token.capitalize()):
	                        wrong_words[token]=1
	                        wrong_count+=1
	                    else:
	                        correct_words.add(token)
	     return wrong_count

	return get_typos_count


def calculate_typo_ratio(df, typo_count_column, token_count_column, typo_ratio_column):
	df[typo_ratio_column]= df[typo_count_column]/len(df[token_count_column])
	return df

def apply_typo_ratio(df):
	get_typos_count = create_get_typos_count()

	df_apply(df,'title_tokens', 'typos_title_count', get_typos_count)
	df_apply(df,'text_tokens', 'typos_text_count', get_typos_count)

	calculate_typo_ratio(df, 'typos_title_count','title_tokens', 'title_typo_ratio')
	calculate_typo_ratio(df, 'typos_text_count','text_tokens', 'text_typo_ratio')
	return df


