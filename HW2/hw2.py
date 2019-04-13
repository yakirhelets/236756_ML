import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

# Mandatory assignment:
# Load the Election Challenge data from the ElectionsData.csv
file_name = 'ElectionsData.csv'
data_frame = pd.read_csv(file_name)

# Identify and set the correct type of each attribute



# lst = list(data_frame._get_axis(1))
# print(lst)
# data_frame.astype()
# print(data_frame.sample(1).dtypes)

# Split the data to train, test, validation sets
labels = data_frame.iloc[:, 0]  # labels
data = data_frame.iloc[:, 1:]  # data
X_train, X_test, Y_train, Y_test =\
    train_test_split(data, labels, train_size=0.9, test_size=0.1, shuffle=False,
                     random_state=None, stratify=None) #todo: changes train size and shuffle
X_train, X_validation, Y_train, Y_validation =\
    train_test_split(X_train, Y_train, train_size=0.83, test_size=0.17, shuffle=False,
                     random_state=None ,stratify=None) #todo: changes train size and shuffle

X_train_prep = X_train.copy()
X_validation_prep = X_validation.copy()
Y_train_prep = Y_train.copy()
Y_validation_prep = Y_validation.copy()
X_test_prep = X_test.copy()
Y_test_prep = Y_test.copy()

# Data preparation actions on the training set:
# Imputation
mod = stats.mode(X_train_prep["Occupation_Satisfaction"])
print(mod[0][0])
xx = X_train_prep["Occupation_Satisfaction"].fillna(mod[0][0])
print(xx)

# Data cleansing
# Normalization
# Feature selection



# Save the 3x2 data sets in CSV files
