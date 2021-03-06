
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib


def get_x(df):
	return df[['title_clean', 'text_clean','text_typo_ratio']]

def get_y(df):
	return df['score']

def split_train_test_data(df):
	x_train, x_test, y_train, y_test = train_test_split(get_x(df), get_y(df),random_state=0,test_size=0.3)

	return x_train, x_test, y_train, y_test

def get_preprocessor():
	preprocessor = ColumnTransformer([
    	('vectorizer_title', CountVectorizer(), 'title_clean'),
    	('vectorizer_text', CountVectorizer(), 'text_clean'),
	])

	return preprocessor

# def calculate_best_param(x_train, y_train):
# 	preprocessor = get_preprocessor()

# 	final_pipe = Pipeline([
# 	    ('preprocessing', preprocessor),
# 	    ('Logistic', LogisticRegression())])

# 	parameters = {
#     'Logistic__solver': ('newton-cg', 'lbfgs', 'sag'),
#     'Logistic__C': ([0.2, 0.4, 0.6, 0.8, 1.0])
# 	}

# 	grid_search = GridSearchCV(final_pipe,
# 	                           parameters,
# 	                           scoring = ["f1", "accuracy", "recall"], 
# 	                           refit= "accuracy",
# 	                           cv=3,
# 	                           verbose = 0)

# 	grid_search.fit(x_train, y_train)


# 	return grid_search.best_params_['Logistic__solver'],  grid_search.best_params_['Logistic__C']


def fit_model(x_train, y_train):
	#best_solver, best_c = calculate_best_param(x_train, y_train)
	preprocessor = get_preprocessor()

	final_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('Logistic', LogisticRegression(solver = 'newton-cg', C = '0.4' ))])

	final_pipe.fit(x_train, y_train)

	return final_pipe


def save_model(pipeline, file_name): 
	joblib.dump(pipeline, file_name)










