import pickle
from typing import *
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error

def eval(model, features: List[int], targets: List[List[int]]) -> int:
    
    preds = []
    for month in features:
        preds.append(model.predict(month - 1))
    return mean_squared_error(targets, preds)


if __name__ == "__main__":
    #load model and data
    linear = pickle.load(open('models/mean_avg.sav', 'rb'))
    
    test_data = pd.read_csv('data/test.csv')
    test_feats = test_data['month'].values
    test_targets = test_data[['standardized_sales', 'standardized_traffic']].values

    #sanity check
    train_data = pd.read_csv('data/train.csv')
    train_feats = train_data['month'].values
    train_targets = train_data[['standardized_sales', 'standardized_traffic']].values
    #
    linear_accuracy = eval(linear, train_feats, train_targets)
    print(linear_accuracy)

    linear_accuracy = eval(linear, test_feats, test_targets)
    print(linear_accuracy)
    
