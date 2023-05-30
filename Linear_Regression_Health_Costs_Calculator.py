import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/insurance.csv')

X = pd.get_dummies(df[['age', 'bmi', 'children','sex','smoker','region']])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

new_data = pd.DataFrame({'age': [1], 
                         'sex': [1], 
                         'bmi': [1], 
                         'children': [1],  
                         'smoker': [1],
                         'region': [1]})

new_data_encoded = pd.get_dummies(new_data)
missing_columns_train = set(X_train.columns) - set(new_data_encoded.columns)
missing_columns_new = set(new_data_encoded.columns) - set(X_train.columns)
for column in missing_columns_train:
    new_data_encoded[column] = 0

for column in missing_columns_new:
    del new_data_encoded[column]

new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

predicted_costs = model.predict(new_data_encoded)
print("Predicted health costs:", predicted_costs)










