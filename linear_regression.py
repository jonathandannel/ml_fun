import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from boston_dataset import BostonDataset

df = BostonDataset().df

x = df['RM'].values.reshape(-1, 1)
# [[val1], [val2], [val3]...]

# Dependent variable, target
y = df['MEDV'].values

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)

# Visualize
plt.figure(figsize=(10, 5))
sns.regplot(x, y)
plt.xlabel('avg rooms per dwelling')
plt.ylabel('median val of homes in $1000s')
plt.show()

prediction = model.predict(np.array([5]).reshape(1, -1))
print(prediction)

