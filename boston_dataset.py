import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class BostonDataset:
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

    def __init__(self):
        pd.options.display.float_format = "{:,.2f}".format
        self.df = pd.read_csv("data/housing.data", 
                            delim_whitespace=True,
                            header=None)
        self.df.columns = self.column_names

    def get_corr_heatmap(self, arr=[]):
        plt.figure(figsize=(16, 10))
        if not arr:
            sns.heatmap(self.df.corr(), annot=True)
        else:
            sns.heatmap(self.df[arr].corr(), annot=True)
        plt.show()
    def get_regplot(self, x, y):
        plt.figure(figsize=(10, 5))
        sns.regplot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    