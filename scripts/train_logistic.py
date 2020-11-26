from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle,os

path = os.path.join('..','raw_data','logistic_model.pkl')

def get_x(df):
	return df[['title_clean', 'text_clean','title_length_char','title_Upper_Ratio','text_stop_words_ratio']]

def get_y(df):
	return df['score']

def split_train_test_data(df):
	x_train, x_test, y_train, y_test = train_test_split(get_x(df), get_y(df),random_state=0,test_size=0.3)

	return x_train, x_test, y_train, y_test

def get_preprocessor():
	preprocessor = ColumnTransformer([
    	('vectorizer_title', CountVectorizer(), 'title_clean'),
    	('vectorizer_text', CountVectorizer(), 'text_clean'),
        ('scaling_title_char', MinMaxScaler(), ['title_length_char'])
	])

	return preprocessor

def fit_model(x_train, y_train):
	#best_solver, best_c = calculate_best_param(x_train, y_train)
	preprocessor = get_preprocessor()

	final_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('Logistic', LogisticRegression(solver = 'newton-cg', C = 0.4 ))])

	final_pipe.fit(x_train, y_train)

	return final_pipe


def save_model(pipeline):
	with open(path, "wb") as file:
            pickle.dump(pipeline, file)
