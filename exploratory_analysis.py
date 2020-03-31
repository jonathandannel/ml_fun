import numpy as numpy
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

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

#################################

HEAD = df.head()
DESCRIPTION = df.describe()
print(HEAD)
print(DESCRIPTION)

#################################

# Visualize data in pairplots
# col_study = ["B", "CRIM" ]
# plt.figure(figsize=(1, 1))
# sns.pairplot(df[col_study], height=2.5)
# plt.show()

#################################
print(df.corr())

# Perform correlation analysis with a heatmap
# Limit correlation to a few relevant fields

col_study = ["ZN", "CRIM", "CHAS", "INDUS", "MEDV" ]
plt.figure(figsize=(16,10))
sns.heatmap(df[col_study].corr(), annot=True)
plt.show()