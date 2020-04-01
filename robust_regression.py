import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from boston_dataset import BostonDataset

boston = BostonDataset()
RANSAC = RANSACRegressor()

df = boston.df

# Feature matrix, target vector
X = df['RM'].values.reshape(-1, 1)
y = df['MEDV'].values 

# Fit model
RANSAC.fit(X, y)

# Outlier
inlier_mask = RANSAC.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

def test():
    print(inlier_mask)
    print(outlier_mask)
    # boston.get_corr_heatmap(['MEDV', 'RM'])
    boston.get_regplot(X, y)

test()






