import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from HW3 import data_preparation
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

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


def showDiagram(parameters_array, score_array, title, parameters, color):
    fig, ax = plt.subplots()
    width = 0.75
    ind = np.arange(len(score_array))
    ax.barh(ind, score_array, width, color=color)
    ax.set_yticks(ind + width/2)
    ax.set_yticklabels(parameters_array, minor=False)
    plt.title(title)
    plt.xlabel('score')
    plt.ylabel(parameters)
    for j, v in enumerate(score_array):
        plt.text(v, j, " "+str(round(v, 2)), color=color, va='center')
    plt.show()

def k_fold_cv_score(KNN_classifier, X_train, Y_train):
    # KFCV
    cross_val_scores = cross_val_score(KNN_classifier, X_train, Y_train, cv=k_folds, scoring='accuracy')
    # avergae score of classifier
    return cross_val_scores.mean()

def clear_arrays():
    score_array.clear()
    parameters_array.clear()

def printTree(DT_classifier):
    dt_graph = StringIO()
    export_graphviz(DT_classifier, out_file=dt_graph, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dt_graph.getvalue())
    Image(graph.create_png())


# -------------------------------------------------------------------
# ------------------------ 0: Prepare the data ----------------------
# -------------------------------------------------------------------

# TODO: uncomment
# data_preparation.prepare_data()
# TODO: make sure data is random

# -------------------------------------------------------------------
# ------------------------ 1: Load the training set -----------------
# -------------------------------------------------------------------

X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file, header=None, names=['Vote'])


# -------------------------------------------------------------------
# ------------------------ 2: Train models --------------------------
# -------------------------------------------------------------------

k_folds = 10

# ------------------------ 2A: K-Nearest Neighbours -----------------

score_array = []
parameters_array = []

# KNN

for i in (1, 3, 5, 7, 9, 13, 15):
    parameters_array.append(i)
    KNN_classifier = KNeighborsClassifier(n_neighbors=i)
    score = k_fold_cv_score(KNN_classifier, X_train, Y_train)
    score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - KNN - K-fold CV', parameters="K value", color='green')

# ------------------------ 2B: Decision Trees ------------------------

clear_arrays()

# Decision Tree Classifier
for cr in ("gini", "entropy"):
    for min_sam in (1.0, 2, 3):
        parameters_array.append((cr, min_sam))

        DT_classifier = DecisionTreeClassifier(criterion=cr, min_samples_split=min_sam)
        score = k_fold_cv_score(DT_classifier, X_train, Y_train)
        score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - Decision Tree - K-fold CV', parameters="criterion, min_samples_split", color='green')

clear_arrays()

# Random Forest
for n_est in (3, 10, 13):
    for max_d in (3, 5, None):
        parameters_array.append((n_est, max_d))
        RF_classifier = RandomForestClassifier(n_estimators=n_est, max_depth=max_d)
        score = k_fold_cv_score(RF_classifier, X_train, Y_train)
        score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - Random Forest - K-fold CV', parameters="n_estimators, max_depth", color='green')

# ------------------------ 2C: Linear Models -------------------------

clear_arrays()

# Perceptron
for max_it in (10, 50, 100, 200, 500, 1000, 2000):
    parameters_array.append(max_it)
    Per_classifier = Perceptron(max_iter=max_it)
    score = k_fold_cv_score(Per_classifier, X_train, Y_train)
    score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - Perceptron - K-fold CV', parameters="max_iter", color='green')

# LinearSVC
# SVC_classifier = LinearSVC(tol=1e-5, random_state=0)
# SVC_classifier = SVC_classifier.fit(X_train, Y_train)

# # LMS
#
# ------------------------ 2D: Naive Bayes ---------------------------

clear_arrays()

# GaussianNB
for var_sm in (1e-1, 1e-3, 1e-5, 1e-9, 1e-11, 1e-13, 1e-15):
    parameters_array.append(var_sm)
    GNB_classifier = GaussianNB(priors=None, var_smoothing=var_sm)
    score = k_fold_cv_score(GNB_classifier, X_train, Y_train)
    score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - Gaussian Naive Bayes - K-fold CV', parameters="var_smoothing", color='green')

clear_arrays()

# GaussianNB
for var_sm in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
    parameters_array.append(var_sm)
    GNB_classifier = GaussianNB(priors=None, var_smoothing=var_sm)
    score = k_fold_cv_score(GNB_classifier, X_train, Y_train)
    score_array.append(score)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='Average score - Gaussian Naive Bayes - K-fold CV', parameters="var_smoothing", color='green')

#
# -------------------------------------------------------------------
# ------------------------ 3: Load the validation set ---------------
# -------------------------------------------------------------------

X_validation_file = 'X_validation.csv'
X_validation = pd.read_csv(X_validation_file)

Y_validation_file = 'Y_validation.csv'
Y_validation = pd.read_csv(Y_validation_file, header=None, names=['Vote'])


# -------------------------------------------------------------------
# ------------------------ 4: Hyper-Parameters Tuning ---------------
# -------------------------------------------------------------------

# KNN
clear_arrays()

for i in (1, 13, 15):
    parameters_array.append(i)

    KNN_classifier = KNeighborsClassifier(n_neighbors=i)
    KNN_classifier = KNN_classifier.fit(X_train, Y_train)
    KNN_prediction = KNN_classifier.predict(X_validation)
    f1 = f1_score(Y_validation, KNN_prediction, average='micro')

    score_array.append(f1)
# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - KNN - Validation set', parameters="K value", color='brown')

clear_arrays()

# Decision Tree Classifier
for cr in ("gini", "entropy"):
    for min_sam in (2, 3):
        parameters_array.append((cr, min_sam))

        DT_classifier = DecisionTreeClassifier(criterion=cr, min_samples_split=min_sam)
        DT_classifier = DT_classifier.fit(X_train, Y_train)
        DT_prediction = DT_classifier.predict(X_validation)
        f1 = f1_score(Y_validation, DT_prediction, average='micro')

        score_array.append(f1)

# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - Decision Tree - Validation set', parameters="criterion, min_samples_split", color='brown')


clear_arrays()

# Random Forest
for n_est in (3, 10, 13):
    parameters_array.append((n_est, None))

    RF_classifier = RandomForestClassifier(n_estimators=n_est, max_depth=None)
    RF_classifier = RF_classifier.fit(X_train, Y_train)
    RF_prediction = RF_classifier.predict(X_validation)
    f1 = f1_score(Y_validation, RF_prediction, average='micro')

    score_array.append(f1)

# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - Random Forest - Validation set', parameters="n_estimators, max_depth", color='brown')

clear_arrays()

# Perceptron
for max_it in (100, 500, 2000):
    parameters_array.append(max_it)

    Per_classifier = Perceptron(max_iter=max_it)
    Per_classifier = Per_classifier.fit(X_train, Y_train)
    Per_prediction = Per_classifier.predict(X_validation)
    f1 = f1_score(Y_validation, Per_prediction, average='micro')

    score_array.append(f1)

# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - Perceptron - Validation set', parameters="max_iter", color='brown')

clear_arrays()

# GaussianNB
for var_sm in (1e-9, 1e-10, 1e-11):
    parameters_array.append(var_sm)

    GNB_classifier = GaussianNB(priors=None, var_smoothing=var_sm)
    GNB_classifier = GNB_classifier.fit(X_train, Y_train)
    GNB_prediction = GNB_classifier.predict(X_validation)
    f1 = f1_score(Y_validation, GNB_prediction, average='micro')

    score_array.append(f1)

# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - Gaussian Naive Bayes - Validation set', parameters="var_smoothing", color='brown')

clear_arrays()

# Final Diagram before selecting the best model

parameters_array.append(('KNN(k=1)'))
KNN_classifier = KNeighborsClassifier(n_neighbors=1)
KNN_classifier = KNN_classifier.fit(X_train, Y_train)
KNN_prediction = KNN_classifier.predict(X_validation)
f1 = f1_score(Y_validation, KNN_prediction, average='micro')
score_array.append(f1)

parameters_array.append('DTree(crit=ent, min_samp=3')
DT_classifier = DecisionTreeClassifier(criterion='entropy', min_samples_split=3)
DT_classifier = DT_classifier.fit(X_train, Y_train)
DT_prediction = DT_classifier.predict(X_validation)
f1 = f1_score(Y_validation, DT_prediction, average='micro')
score_array.append(f1)

printTree(DT_classifier)

parameters_array.append('RandF(n_est=13, max_d=None')
RF_classifier = RandomForestClassifier(n_estimators=13, max_depth=None)
RF_classifier = RF_classifier.fit(X_train, Y_train)
RF_prediction = RF_classifier.predict(X_validation)
f1 = f1_score(Y_validation, RF_prediction, average='micro')
score_array.append(f1)

parameters_array.append('Percep(max_iter=2000)')
Per_classifier = Perceptron(max_iter=2000)
Per_classifier = Per_classifier.fit(X_train, Y_train)
Per_prediction = Per_classifier.predict(X_validation)
f1 = f1_score(Y_validation, Per_prediction, average='micro')
score_array.append(f1)

parameters_array.append('GaussNB(var_smooth=1e-9')
GNB_classifier = GaussianNB(priors=None, var_smoothing=1e-9)
GNB_classifier = GNB_classifier.fit(X_train, Y_train)
GNB_prediction = GNB_classifier.predict(X_validation)
f1 = f1_score(Y_validation, GNB_prediction, average='micro')
score_array.append(f1)

# produce diagram with the parameters and score
showDiagram(parameters_array, score_array, title='F1 score - Combined - Validation set', parameters="parameters", color='orange')



# -------------------------------------------------------------------
# ------------------------ 5: Best Model Selection ------------------
# -------------------------------------------------------------------

RF_classifier = RandomForestClassifier(n_estimators=13, max_depth=None)
RF_classifier = RF_classifier.fit(X_train, Y_train)

# -------------------------------------------------------------------
# ------------------------ 6: Predictions ---------------------------
# -------------------------------------------------------------------

# Predictions

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)


Y_test_file = 'Y_test.csv'
Y_test = pd.read_csv(Y_test_file)


# ------------------------------ 6A ---------------------------------

# The selected classifier with the selected parameters
selected_classifier = RF_classifier
selected_classifier_prediction = selected_classifier.predict(X_test)
selected_classifier_prediction.to_csv("prediction_results.csv")

# ------------------------------ 6B ---------------------------------
# The party that will win the majority of votes

printHistogram(selected_classifier_prediction)

# # ------------------------------ 6C ---------------------------------
# # Probable voters per party
#
# # ------------------------------ 6D ---------------------------------
# # Factor which by manipulating we can change the winning party
#
# # ------------------------------ 6E ---------------------------------
# # Confusion matrix and Overall test error
#
# # TODO: uncomment confusion matrix code below
# # confusion_matrix = confusion_matrix(Y_test, selected_classifier_prediction)
# # confusion_matrix_DF = pd.DataFrame(confusion_matrix, columns=['Predicted Pos', 'Predicted Neg'], index=['Actual Pos', 'Actual Neg'])
# # print(confusion_matrix_DF)
