import pandas as pd
from cleaning_logistic import apply_cleaning
from typo_func import apply_typo_ratio
from train_logistic import split_train_test_data, fit_model, save_model


data = pd.read_csv('../raw_data/working.csv')

apply_cleaning(data)

apply_typo_ratio(data)



x_train, x_test, y_train, y_test = split_train_test_data(data)

pipe_logistic = fit_model(x_train, y_train)

save_model(pipe_logistic)

print(pipe_logistic.score(x_test, y_test))

