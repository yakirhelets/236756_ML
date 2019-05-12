import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import LocalOutlierFactor
from . import data_preparation


# -------------------------------------------------------------------
# ------------------------ 1: Prepare the data ----------------------
# -------------------------------------------------------------------

data_preparation.prepare_data()


# -------------------------------------------------------------------
# ------------------------ 2: Load the train set --------------------
# -------------------------------------------------------------------

X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file)


# -------------------------------------------------------------------
# ------------------------ 3: Train models --------------------------
# -------------------------------------------------------------------

# ------------------------ 3A: KNN ----------------------------------

KNN_classifier = KNeighborsClassifier(n_neighbors=1)
KNN_classifier.fit(X_train, Y_train)

# ------------------------ 3B: Decision Tree ------------------------

DT_classifier = tree.DecisionTreeClassifier()
DT_classifier = DT_classifier.fit(X_train, Y_train)

# ------------------------ 3C: Linear Models -------------------------

# Perceptron
Per_classifier = Perceptron(tol=1e-3, random_state=0)
Per_classifier = Per_classifier.fit(X_train, Y_train)

# LinearSVC
SVC_classifier = LinearSVC(tol=1e-5, random_state=0)
SVC_classifier = SVC_classifier.fit(X_train, Y_train)

# LMS

# ------------------------ 3D: Naive Bayes ---------------------------

# GaussianNB
GNB_classifier = GaussianNB()
GNB_classifier = GNB_classifier.fit(X_train, Y_train)


# -------------------------------------------------------------------
# ------------------------ 4: Load the validation set ---------------
# -------------------------------------------------------------------

X_validation_file = 'X_validation.csv'
X_validation = pd.read_csv(X_validation_file)





# Predictions

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)


KNN_prediction = KNN_classifier.predict(X_test)
DT_prediction = DT_classifier.predict(X_test)
Per_prediction = Per_classifier.predict(X_test)
SVC_prediction = SVC_classifier.predict(X_test)
GNB_prediction = GNB_classifier.predict(X_test)



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
