import pandas as pd
from preparing_df import apply_preparing, apply_preparing_merge
from cleaning import apply_cleaning
from typo import apply_typo_ratio
from train import split_train_test_data, fit_model, save_model


true = pd.read_csv('../raw_data/True.csv')
fake = pd.read_csv('../raw_data/Fake.csv')

data = apply_preparing(true,fake)
data = apply_preparing_merge(data)

apply_cleaning(data)

apply_typo_ratio(data)



x_train, x_test, y_train, y_test = split_train_test_data(data)

pipe = fit_model(x_train, y_train)

save_model(pipe, 'model_test.joblib')

print(pipe.score(x_test, y_test))
