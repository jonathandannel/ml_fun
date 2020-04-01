import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from boston_dataset import BostonDataset

boston = BostonDataset()

df = boston.df

rm_values = df['RM'].values.reshape(-1, 1)

# boston.get_corr_heatmap(['MEDV', 'LSTAT', 'INDUS', 'B'])




