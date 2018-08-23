import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

os.getcwd()
"""
https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
Predict prices of car based on the attributes
"""
indt = pd.read_csv("datasets/imports85.data", header=None)
indt.columns = ['symboling', 'norm_losses', 'make', 'fuel', 'aspiration', 'door', 'body', 'drive', 'engine_loc', 'wheel_base', 'length', 'width', 'height', 'curb', 'enginetype', 'cylinders', 'engsize', 'fuelsys', 'bore', 'stroke',  'compression', 'hp', 'peakrpm', 'citympg', 'hwympg', 'price']
target = ['price']
catvar = ['make', 'fuel', 'aspiration', 'door', 'body', 'drive', 'engine_loc', 'enginetype', 'cylinders', 'fuelsys']
contvar = ['symboling', 'norm_losses', 'wheel_base', 'length', 'width', 'height', 'curb', 'engsize', 'bore', 'stroke',  'compression', 'hp', 'peakrpm', 'citympg', 'hwympg']

indt = indt[target+catvar+contvar]
indt.dtypes

# Replace all ? by np.nan
indt = indt.replace('?', np.nan)
indt.isnull().sum()

# Correct for all the column formats

for col in indt.columns:
    if col in catvar:
        indt[col] = indt[col].astype(object)
    else:
        indt[col] = indt[col].astype(float)

indt.head(2)

# Replace missing values by mode values
# indt.isnull().any(axis=1).sum()

mode_value = indt[~indt.isnull().any(axis=1)].mode()[0:1]

for col in indt.columns:
    indt.loc[indt[col].isnull(),col] = mode_value[col][0]

indt.isnull().any(axis=1).sum()

# Add the average price of car by fuel type as a vairable
indt.groupby(['fuel', 'aspiration'])['hp'].agg(['mean','count'])
indt['avgPbyF'] = indt.groupby(['fuel', 'aspiration'])['hp'].transform('mean')

indt['avgPbyF'].value_counts()
