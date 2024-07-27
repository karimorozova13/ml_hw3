import pandas as pd

X_train = pd.read_csv('./datasets/mod_04_hw_train_data.csv')
X_test = pd.read_csv('./datasets/mod_04_hw_valid_data.csv')


# %%
y_train = X_train['Salary']
y_test = X_test['Salary']

# %%

X_train = X_train.drop(['Name', 'Phone_Number', 'Salary', 'Date_Of_Birth', 'Qualification'], axis=1)
X_test = X_test.drop(['Name', 'Phone_Number', 'Salary','Date_Of_Birth', 'Qualification'], axis=1)

# %%
X_train_cat = X_train.select_dtypes(include='object')
X_train_num = X_train.select_dtypes(exclude='object')

# %%
X_test_cat = X_test.select_dtypes(include='object')
X_test_num = X_test.select_dtypes(exclude='object')

# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')

X_train_num = imputer.fit_transform(X_train_num)
X_test_num = imputer.transform(X_test_num)

# %%
from sklearn.preprocessing import PowerTransformer 

transformer = PowerTransformer().set_output(transform='pandas')

X_train_num = transformer.fit_transform(X_train_num)
X_test_num = transformer.transform(X_test_num)

# %%
from sklearn.preprocessing import OneHotEncoder 

encoder = (OneHotEncoder(drop='if_binary',
                        sparse_output=False,)
          .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat, y_train)
X_test_cat = encoder.transform(X_test_cat)

# %%
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(pd.concat([X_train_cat, X_train_num], axis=1), y_train)

y_pred = regressor.predict(pd.concat([X_test_cat, X_test_num], axis=1))

# %%
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Validation MAPE: {mape:.2%}')

