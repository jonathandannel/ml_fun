import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from boston_df import BostonDF

boston = BostonDF()

df = boston.df

boston.get_corr_heatmap(['MEDV', 'LSTAT', 'INDUS', 'B'])