import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from HW5 import data_preparation
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# -------------------------------------------------------------------
# ------------------------ *: Helper functions ----------------------
# -------------------------------------------------------------------

def printHistogram(dataframe, vote_column_name):
    df = dataframe[vote_column_name]
    df = df.value_counts()
    df_idx = list(df.index)
    print(df_idx)
    df_vals = list(df.values)
    print(df_vals / sum(df_vals))
    y_pos = np.arange(len(df_vals))
    plt.bar(y_pos, df_vals, align='center')
    plt.xticks(y_pos, df_idx, fontsize=7)
    plt.show()

# -------------------------------------------------------------------
# ---------------- Loading and Data preparation ---------------------
# -------------------------------------------------------------------

data_preparation.prepare_data()

# -------------------------------------------------------------------
# ------------------------ Loading the sets -------------------------
# -------------------------------------------------------------------

# Train sets load
X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file, header=None, names=['Vote'])

# Validation sets load
X_validation_file = 'X_validation.csv'
X_validation = pd.read_csv(X_validation_file)

Y_validation_file = 'Y_validation.csv'
Y_validation = pd.read_csv(Y_validation_file, header=None, names=['Vote'])

# Test sets load
X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)

Y_test_file = 'Y_test.csv'
Y_test = pd.read_csv(Y_test_file, header=None, names=['Vote'])

# New data (prepared) load
X_new_data_file = 'X_new_data.csv'
X_new_data = pd.read_csv(X_new_data_file)

# -------------------------------------------------------------------
# ---------------- 1st process: Train and predict -------------------
# -------------------------------------------------------------------

# train on both the train and the validation set

X_train_and_valid = pd.concat([X_train, X_validation], ignore_index=True)
Y_train_and_valid = pd.concat([Y_train, Y_validation], ignore_index=True)

RF_classifier = RandomForestClassifier(n_estimators=13, max_depth=None)
RF_classifier = RF_classifier.fit(X_train_and_valid, Y_train_and_valid)

# predict on the test set

RF_prediction = RF_classifier.predict(X_test)

Y_test_as_array = []
for i in range(len(Y_test.values)):
    Y_test_as_array.append(Y_test.values[i][0])

Y_test_as_array = np.array(Y_test_as_array)

# get results and evaluate perf on the test set

# TODO do CV

f1 = f1_score(Y_test, RF_prediction, average='micro')
print("F1: " + str(f1*100))
accuracy = np.sum(Y_test_as_array == RF_prediction) / len(Y_test_as_array)
print("Accuracy: " + str(accuracy*100))
print("Error: " + str((1-accuracy)*100))
print("Precision: " + str(precision_score(Y_test, RF_prediction, average='micro')))
print("Recall: " + str(recall_score(Y_test, RF_prediction, average='micro')))

# before predicting the new data - train on the entire original data

combined_X = pd.concat([X_train, X_validation, X_test], ignore_index=True)
combined_Y = pd.concat([Y_train, Y_validation, Y_test], ignore_index=True)

RF_classifier_for_new_data = RandomForestClassifier(n_estimators=13, max_depth=None)
RF_classifier_for_new_data = RF_classifier_for_new_data.fit(combined_X, combined_Y)

identity_card_col = X_new_data['IdentityCard_Num']
X_new_data = X_new_data.drop(['IdentityCard_Num'], axis=1)

prediction = pd.DataFrame(RF_classifier_for_new_data.predict(X_new_data))
prediction.columns = ['PredictVote']

# edit final_predictions so that it has two columns as required
final_prediction = pd.concat([identity_card_col, prediction], axis=1)

final_prediction.to_csv("Final_Predictions.csv")

# ---------------------------- Task 1 -------------------------------

# Predict which party would win the majority of votes

printHistogram(final_prediction, 'PredictVote')
# TODO check why different runs produce different results

# histogram of the distribution of votes on the original dataset

printHistogram(combined_Y, 'Vote')

# ---------------------------- Task 2 -------------------------------

# Predict the division of votes between the various parties



# ---------------------------- Task 3 -------------------------------

# Predict the vote of each voter in the new sample





# -------------------------------------------------------------------
# ---------------- 2nd process: Identify coalition ------------------
# -------------------------------------------------------------------



# ---------------------------- Task 4 -------------------------------

# Identify a steady coalition

