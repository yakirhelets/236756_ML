import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from . import data_preparation

# -------------------------------------------------------------------
# ------------------------ 0: Prepare the data ----------------------
# -------------------------------------------------------------------

data_preparation.prepare_data()


# -------------------------------------------------------------------
# ------------------------ 1: Load the training set -----------------
# -------------------------------------------------------------------

X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file)


# -------------------------------------------------------------------
# ------------------------ 2: Train models --------------------------
# -------------------------------------------------------------------

# ------------------------ 2A: K-Nearest Neighbours -----------------

# KNN
KNN_classifier = KNeighborsClassifier(n_neighbors=1)
KNN_classifier.fit(X_train, Y_train)

# ------------------------ 2B: Decision Trees ------------------------

# Decision Tree Classifier
DT_classifier = tree.DecisionTreeClassifier()
DT_classifier = DT_classifier.fit(X_train, Y_train)

# Random Forest
RF_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF_classifier = RF_classifier.fit(X_train, Y_train)

# ------------------------ 2C: Linear Models -------------------------

# Perceptron
Per_classifier = Perceptron(tol=1e-3, random_state=0)
Per_classifier = Per_classifier.fit(X_train, Y_train)

# # LinearSVC
# SVC_classifier = LinearSVC(tol=1e-5, random_state=0)
# SVC_classifier = SVC_classifier.fit(X_train, Y_train)

# LMS

# ------------------------ 2D: Naive Bayes ---------------------------

# GaussianNB
GNB_classifier = GaussianNB()
GNB_classifier = GNB_classifier.fit(X_train, Y_train)


# -------------------------------------------------------------------
# ------------------------ 3: Load the validation set ---------------
# -------------------------------------------------------------------

X_validation_file = 'X_validation.csv'
X_validation = pd.read_csv(X_validation_file)

Y_validation_file = 'Y_validation.csv'
Y_validation = pd.read_csv(Y_validation_file)



# -------------------------------------------------------------------
# ------------------------ 4: Hyper-Parameters Tuning ---------------
# -------------------------------------------------------------------

# TODO: determine where to insert cross validation
# KNN_prediction = KNN_classifier.predict(X_test)
# DT_prediction = DT_classifier.predict(X_test)
# RF_prediction = RF_classifier.predict(X_test)
# Per_prediction = Per_classifier.predict(X_test)
# SVC_prediction = SVC_classifier.predict(X_test)
# GNB_prediction = GNB_classifier.predict(X_test)

# -------------------------------------------------------------------
# ------------------------ 5: Best Model Selection ------------------
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# ------------------------ 6: Predictions ---------------------------
# -------------------------------------------------------------------

# Predictions

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)



# ------------------------------ 6A ---------------------------------

# selected_classifier = ...
# selected_classifier_prediction = selected_classifier.predict(X_test)
# selected_classifier_prediction.to_csv("prediction_results.csv")

# ------------------------------ 6B ---------------------------------
# The party that will win the majority of votes
# TODO: use mode on "selected_classifier_prediction"

# ------------------------------ 6C ---------------------------------
# Probable voters per party

# ------------------------------ 6D ---------------------------------
# Factor which by manipulating we can change the winning party

# ------------------------------ 6E ---------------------------------
# Confusion matrix and Overall test error


# -------------------------------------------------------------------
# ------------------------ *: Helper functions ----------------------
# -------------------------------------------------------------------


def printHistogram(dataframe):
    # vote_values = list(set(dataframe['Vote']))
    df = dataframe['Vote']
    df = df.value_counts()
    df_idx = list(df.index)
    df_vals = list(df.values)
    y_pos = np.arange(len(df_vals))
    plt.bar(y_pos, df_vals, align='center')
    plt.xticks(y_pos, df_idx)
    plt.show()


# Usage example:
printHistogram(pd.read_csv('ElectionsData.csv'))
