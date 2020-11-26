import pandas as pd
from scripts.cleaning import apply_cleaning
from scripts.typo_func import apply_typo_ratio
from scripts.train_logistic import split_train_test_data, fit_model, save_model


data = pd.read_csv('../raw_data/working.csv')

data_clean = apply_cleaning(data)

#data_full = apply_typo_ratio(data_clean)



x_train, x_test, y_train, y_test = split_train_test_data(data_clean)

pipe_logistic = fit_model(x_train, y_train)

save_model(pipe_logistic)

print(pipe_logistic.score(x_test, y_test))

