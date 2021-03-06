import pandas as pd
from preparing_df import apply_preparing 
from cleaning import apply_cleaning
from typo import apply_typo_ratio
from train import split_train_test_data, fit_model, save_model


true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

data = apply_preparing(true,fake)

apply_cleaning(data)

apply_typo_ratio(data)



x_train, x_test, y_train, y_test = split_train_test_data(data)

pipe = fit_model(x_train, y_train)

save_model(pipe, 'model_test')

print(pipe.score(x_test, y_test))

