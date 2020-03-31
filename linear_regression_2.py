import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/housing.data", delim_whitespace=True, header=None)
column_names = [
    'CRIM', # per capita crime rate by town
    'ZN', # proportion of residental land zoned for lots > 25,000 sq. ft.
    'INDUS', # proportion of non-retail business acres per town
    'CHAS', # charles river dummy variable: 1 if tract bounds river, 0 otherwise
    'NOX', # nitric oxides concentration (parts per 10 million)
    'RM', # average number of rooms per dwelling
    'AGE', # proportion of owner-occupied units built prior to 1940
    'DIS', # weighted distances to five Boston employment centres
    'RAD', # index of accessibility to radial highways
    'TAX', # full-value property-tax rate per $10,000
    'PTRATIO', # pupil-teacher ratio by town
    'B', # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    'LSTAT', # % lower status of population
    'MEDV' # median value of owner-occupied homes in 1000s
]

df.columns = column_names
pd.options.display.float_format = "{:,.2f}".format

### 1: Arrange data into features matrix and target vector

# Features matrix
x = df['LSTAT'].values.reshape(-1, 1)
print(x)

# Target vector. Do not need to reshape.
y = df['MEDV'].values
print(y)

### 2: Fit the model to the data
model = LinearRegression()
model.fit(x, y)


### 3: Apply the model to new data
prediction = model.predict(np.array([5]).reshape(1, -1))
print(prediction)

