import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from HW5 import data_preparation, automatic_model_selection
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score

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

def get_scores(Y, prediction, clf_name):
    Y_as_array = []
    for i in range(len(Y.values)):
        Y_as_array.append(Y.values[i][0])

    Y_as_array = np.array(Y_as_array)

    # get results and evaluate perf on the test set

    f1 = f1_score(Y, prediction, average='micro')
    print(str(clf_name) + " F1: " + str(f1 * 100))
    accuracy = np.sum(Y_as_array == prediction) / len(Y_as_array)
    print(str(clf_name) + " Accuracy: " + str(accuracy * 100))
    print(str(clf_name) + " Error: " + str((1 - accuracy) * 100))
    print(str(clf_name) + " Precision: " + str(precision_score(Y, prediction, average='micro')))
    print(str(clf_name) + " Recall: " + str(recall_score(Y, prediction, average='micro')))
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

# train on the train set and test on the validation set

np.random.seed(seed=0)

RF_classifier = RandomForestClassifier(n_estimators=13, max_depth=None, random_state=0)
RF_classifier = RF_classifier.fit(X_train, Y_train)
RF_valid_pred = RF_classifier.predict(X_validation)
get_scores(Y_validation, RF_valid_pred, "RF(valid)")

# train on both the train and the validation set

X_train_and_valid = pd.concat([X_train, X_validation], ignore_index=False)
Y_train_and_valid = pd.concat([Y_train, Y_validation], ignore_index=False)

print(automatic_model_selection.select_automatically(X_train_and_valid, Y_train_and_valid))

RF_classifier = RF_classifier.fit(X_train_and_valid, Y_train_and_valid)
RF_test_pred = RF_classifier.predict(X_test)
get_scores(Y_test, RF_test_pred, "RF(test)")

# Checking the performance of more classifiers:

#  >>>>>>  MLP  <<<<<<

mlp_clf = MLPClassifier()
mlp_clf.fit(X_train_and_valid, Y_train_and_valid)
mlp_pred = mlp_clf.predict(X_test)
get_scores(Y_test, mlp_pred, "MLP(test)")

#  >>>>>>  SVM  <<<<<<

svm_clf = SVC(gamma='auto')
svm_clf.fit(X_train_and_valid, Y_train_and_valid)
svm_pred = svm_clf.predict(X_test)
get_scores(Y_test, svm_pred, "SVM(test)")

# before predicting the new data - train on the entire original data

combined_X = pd.concat([X_train, X_validation, X_test], ignore_index=False)
combined_Y = pd.concat([Y_train, Y_validation, Y_test], ignore_index=False)

# training our selected classifier - random forest - on all of the original data
RF_classifier_for_new_data = RF_classifier.fit(combined_X, combined_Y)

identity_card_col = X_new_data['IdentityCard_Num']

X_new_data = X_new_data.drop(['IdentityCard_Num'], axis=1)
prediction = pd.DataFrame(RF_classifier_for_new_data.predict(X_new_data))
Y_new_data = prediction.copy(deep=True)

prediction.columns = ['PredictVote']

# edit final_predictions so that it has two columns as required
final_prediction = pd.concat([identity_card_col, prediction], axis=1, ignore_index=False)

final_prediction.set_index('IdentityCard_Num', inplace=True)

final_prediction.to_csv("Final_Predictions.csv")

Y_new_data.columns = ['Vote']
X_and_Y_new_data = pd.concat([X_new_data, Y_new_data], axis=1)


# ---------------------------- Tasks 1-2-3 -------------------------------

# (1) Predict which party would win the majority of votes

# histogram of the distribution of votes on the original dataset

printHistogram(combined_Y, 'Vote')

# histogram of the distribution of votes on the new dataset

printHistogram(final_prediction, 'PredictVote')

# (2) Predict the division of votes between the various parties >>>

# **** Percentages are being printed with the histogram ****

# (3) Predict the vote of each voter in the new sample >>>
# ***** Appears in the file Final_Predictions.csv *****


# -------------------------------------------------------------------
# ---------------- 2nd process: Identify coalition ------------------
# -------------------------------------------------------------------

# ---------------------------- Task 4 -------------------------------

# Identify a steady coalition

X_values = X_new_data.values
colors = (0, 0, 1)
area = np.pi

cols_num_wanted = 8

for i in range(1, cols_num_wanted + 1):
    for j in range(i, cols_num_wanted + 1):

        # We don't want a comparison between a feature and itself
        if i == j:
            continue

        kmeans = KMeans(n_clusters=2)
        X_selected_values = X_values[:, [i, j]]
        kmeans.fit(X_selected_values)

        x = X_selected_values[:, 0]
        y = X_selected_values[:, 1]

        # Plot
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], s=200, c='r', marker='s', label='center_1')
        plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], s=200, c='g', marker='s', label='center_2')
        plt.grid()
        plt.legend()

        plt.title(X_new_data.columns[i] + " vs. " + X_new_data.columns[j])
        plt.xlabel(X_new_data.columns[i])
        plt.ylabel(X_new_data.columns[j])
        plt.show()



parties = ["Khakis", "Oranges", "Purples", "Turquoises", "Yellows", "Blues", "Whites",
           "Greens", "Violets", "Browns", "Reds", "Greys", "Pinks"]

print(sorted(parties))

# making the clusters according to the predicted labels
all_data = X_and_Y_new_data

total = all_data.shape[0]

# Percentages of parties among the entire train set

print("Percentages of all parties")
for party in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == party]
    percentage = all_data_copy.shape[0] / total  ## Filling the percentages of voting among the parties
    print("Party: " + str(party) + ", percentage: " + str(percentage))

print("**********")

similarity_threshold = 0.995

parties_voting_percentage = {}

# Division according to Weighted_education_rank

cluster_A_parties_weighted = []
cluster_B_parties_weighted = []

print("Division according to Weighted_education_rank")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
    parties_voting_percentage[i] = all_data_copy.shape[0] / total  ## Filling the percentages of voting among the parties
    cluster_A_vals = all_data_copy[all_data_copy['Weighted_education_rank'] > 0.5]
    cluster_A_percent = cluster_A_vals.shape[0] / total
    cluster_B_percent = 1-cluster_A_percent
    print(i + ": Cluster A = " + str(cluster_A_percent) + ", Cluster B = " + str(cluster_B_percent))
    if cluster_B_percent > similarity_threshold:
        cluster_B_parties_weighted.append(i)
    else:
        cluster_A_parties_weighted.append(i)

print("**********")

# Division according to Avg_monthly_expense_on_pets_or_plants

cluster_A_parties_pets = []
cluster_B_parties_pets = []

print("Division according to Avg_monthly_expense_on_pets_or_plants")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
    cluster_A_vals = all_data_copy[all_data_copy['Avg_monthly_expense_on_pets_or_plants'] > 1]
    cluster_A_percent = cluster_A_vals.shape[0] / total
    cluster_B_percent = 1-cluster_A_percent
    print(i + ": Cluster A = " + str(cluster_A_percent) + ", Cluster B = " + str(1-cluster_A_percent))
    if cluster_B_percent > similarity_threshold:
        cluster_B_parties_pets.append(i)
    else:
        cluster_A_parties_pets.append(i)

print("**********")

# Dividing to coalition and opposition

print(parties_voting_percentage)

# Weighted_education_rank

cluster_A_parties_weighted_percent = 0

for i in cluster_A_parties_weighted:
    cluster_A_parties_weighted_percent = cluster_A_parties_weighted_percent + parties_voting_percentage[i]

print("**********")

print("Cluster A percents (Weighted): " + str(cluster_A_parties_weighted_percent))
print(cluster_A_parties_weighted)
print("Cluster B percents (Weighted): " + str(1-cluster_A_parties_weighted_percent))
print(cluster_B_parties_weighted)

# Avg_monthly_expense_on_pets_or_plants

cluster_A_parties_pets_percent = 0

for i in cluster_A_parties_pets:
    cluster_A_parties_pets_percent = cluster_A_parties_pets_percent + parties_voting_percentage[i]

print("**********")

print("Cluster A percents (Pets or Plants): " + str(cluster_A_parties_pets_percent))
print(cluster_A_parties_pets)
print("Cluster B percents (Pets or Plants): " + str(1-cluster_A_parties_pets_percent))
print(cluster_B_parties_pets)

