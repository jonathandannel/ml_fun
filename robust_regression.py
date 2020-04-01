import numpy as np
import pandas as pd
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

# [boolean]
inlier_mask = RANSAC.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
# [3, 4, 5, 6, 7, 8, 9] rooms

line_y_ransac = RANSAC.predict(line_X.reshape(-1, 1))

def test():
    # print(inlier_mask)
    # print(outlier_mask)
    # boston.get_corr_heatmap(['MEDV', 'RM'])
    boston.get_regplot(X, y)

# Let's illustrate what happened
def plot():
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 8))
    plt.xlabel("rooms")
    plt.ylabel("medv")
    plt.legend(loc="upper left")

    print(line_y_ransac)

    # X = df['RM'].values.reshape(-1, 1)
    # y = df['MEDV'].values 
    
    # inliers for X and y
    plt.scatter(X[inlier_mask], y[inlier_mask],
                c="blue", marker="o", label="inliers")
    
    # outliers for x and y
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c="brown", marker="s", label="outliers")
    
    # Predicted value
    plt.plot(line_X, line_y_ransac, color="red")
    
    plt.show()

plot()




