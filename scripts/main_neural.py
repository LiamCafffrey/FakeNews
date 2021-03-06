import pandas as pd
import scripts
from preparing_neural import apply_preparing_merge
from cleaning import apply_cleaning
from train_neural import split_train_test_data, fit_model, save_model
from preparing_neural import apply_lemmatize
from train_neural import get_preprocessor



data = pd.read_csv('../raw_data/working.csv')

data_clean = apply_cleaning(data)

data_merge = apply_preparing_merge(data_clean)

data_lemmatize = apply_lemmatize(data_merge)

x_train, x_test, y_train, y_test = split_train_test_data(data_lemmatize)


x_train_preprocessed, x_test_preprocessed = get_preprocessor(x_train,x_test)

neural_model = fit_model(x_train_preprocessed, y_train)

save_model(neural_model)

print(neural_model.evaluate(x_test_preprocessed, y_test))
