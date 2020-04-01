import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from boston_dataset import BostonDataset

df = BostonDataset().df

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
prediction = model.predict(np.array([30]).reshape(1, -1))
print(prediction)

