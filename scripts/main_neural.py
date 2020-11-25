import pandas as pd
from preparing_neural import apply_preparing_merge
from cleaning_neural import apply_cleaning
from typo_func import apply_typo_ratio # we need to fix this for deep learning(text+title combined)
from train_neural import split_train_test_data, fit_model, save_model


data = pd.read_csv('../raw_data/working.csv')

data = apply_preparing_merge(data)

apply_cleaning(data)

apply_typo_ratio(data)



x_train, x_test, y_train, y_test = split_train_test_data(data)

pipe_neural = fit_model(x_train, y_train)

save_model(pipe_neural)

print(pipe_neural.score(x_test, y_test))
