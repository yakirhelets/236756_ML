import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from HW3 import data_preparation, automatic_model_selection
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict

# -------------------------------------------------------------------
# ------------------------ *: Helper functions ----------------------
# -------------------------------------------------------------------


def printHistogram(dataframe):
    df = dataframe['Vote']
    df = df.value_counts()
    df_idx = list(df.index)
    df_vals = list(df.values)
    y_pos = np.arange(len(df_vals))
    plt.bar(y_pos, df_vals, align='center')
    plt.xticks(y_pos, df_idx, fontsize=7)
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

        if cr == "entropy" and min_sam == 3:
            tree.export_graphviz(DT_classifier, out_file="Decision_tree.dot")

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

selected_classifier = RandomForestClassifier(n_estimators=13, max_depth=None)

# Combine the train and validation to fit on all of the data (except test)
X_train_and_validation = pd.concat([X_train, X_validation], ignore_index=True)
Y_train_and_validation = pd.concat([Y_train, Y_validation], ignore_index=True)

selected_classifier = selected_classifier.fit(X_train_and_validation, Y_train_and_validation)


# -------------------------------------------------------------------
# ------------------------ 6: Predictions ---------------------------
# -------------------------------------------------------------------

# Predictions

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)


Y_test_file = 'Y_test.csv'
Y_test = pd.read_csv(Y_test_file, header=None, names=['Vote'])


# ------------------------------ 6A ---------------------------------

# The selected classifier with the selected parameters
selected_classifier_prediction = selected_classifier.predict(X_test)
selected_classifier_prediction_df = pd.DataFrame(selected_classifier_prediction)
selected_classifier_prediction_df.to_csv("prediction_results.csv")


Y_test_as_array = []
for i in range(len(Y_test.values)):
    Y_test_as_array.append(Y_test.values[i][0])

Y_test_as_array = np.array(Y_test_as_array)




# ------------------------------ 6B ---------------------------------
# The party that will win the majority of votes

selected_classifier_prediction_df.columns = ['Vote']
printHistogram(selected_classifier_prediction_df)


# Histogram of the predictions on the train_and_validation set
selected_classifier_prediction_train_and_valid = selected_classifier.predict(X_train_and_validation)
selected_classifier_prediction_train_and_valid_df = pd.DataFrame(selected_classifier_prediction_train_and_valid)
Y_train_and_validation_as_array = []
for i in range(len(Y_train_and_validation.values)):
    Y_train_and_validation_as_array.append(Y_train_and_validation.values[i][0])

Y_train_and_validation_as_array = np.array(Y_train_and_validation_as_array)
selected_classifier_prediction_train_and_valid_df.columns = ['Vote']
printHistogram(selected_classifier_prediction_train_and_valid_df)

# ------------------------------ 6C ---------------------------------
# Probable voters per party

voting_threshold = 0.4
classes = selected_classifier.classes_
predictions = selected_classifier.predict_proba(X_test)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions_probabilities.csv")

probable_voters = []

for i in range(len(predictions)):
    probable_classes_per_voter = []
    for j in range(len(predictions[0])):
        if predictions[i][j] > voting_threshold:
            probable_classes_per_voter.append(classes[j])
    probable_voters.append(probable_classes_per_voter)

probable_voters_df = pd.DataFrame(probable_voters)
probable_voters_df.to_csv("probable_voters.csv")

# ------------------------------ 6D ---------------------------------
# Confusion matrix and Overall test error


confusion_mat = confusion_matrix(Y_test_as_array, selected_classifier_prediction, labels=["Khakis", "Oranges", "Purples", "Turquoises",
                                                                                          "Yellows", "Blues", "Whites", "Greens",
                                                                                          "Violets", "Browns", "Reds", "Greys",
                                                                                          "Pinks"])
confusion_mat_DF = pd.DataFrame(confusion_mat, columns=["Predicted Khakis", "Predicted Oranges", "Predicted Purples", "Predicted Turquoises",
                                                          "Predicted Yellows", "Predicted Blues", "Predicted Whites", "Predicted Greens",
                                                          "Predicted Violets", "Predicted Browns", "Predicted Reds", "Predicted Greys",
                                                          "Predicted Pinks"],
                                                index=["Actual Khakis", "Actual Oranges", "Actual Purples", "Actual Turquoises",
                                                       "Actual Yellows", "Actual Blues", "Actual Whites", "Actual Greens",
                                                       "Actual Violets", "Actual Browns", "Actual Reds", "Actual Greys",
                                                       "Actual Pinks"])
confusion_mat_DF.to_csv("confusion_matrix.csv")

f1 = f1_score(Y_test, selected_classifier_prediction_df, average='micro')
print("F1: " + str(f1*100))
accuracy = np.sum(Y_test_as_array == selected_classifier_prediction) / len(Y_test_as_array)
print("Accuracy: " + str(accuracy*100))
print("Error: " + str((1-accuracy)*100))
print("Precision: " + str(precision_score(Y_test, selected_classifier_prediction, average='micro')))
print("Recall: " + str(recall_score(Y_test, selected_classifier_prediction, average='micro')))


# -------------------------------------------------------------------
# ------------------------ 7: Bonus Part ----------------------------
# -------------------------------------------------------------------

# ------------------------------ 7A ---------------------------------
# Automation of the model selection procedure

print(automatic_model_selection.select_automatically(X_train, Y_train))


# ------------------------------ 7C ---------------------------------
# Factor which by manipulating we can change the winning party

# ******** CODE IS IN data_preparation.py file at the end of section 2