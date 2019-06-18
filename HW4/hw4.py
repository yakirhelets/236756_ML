import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# -------------------------------------------------------------------
# ------------------------ *: Helper functions ----------------------
# -------------------------------------------------------------------


def printResults(classifier, X_test_set, Y_test_set, X_train_set, Y_train_set, classifier_name):
    Y_test_as_array = []
    for i in range(len(Y_test_set.values)):
        Y_test_as_array.append(Y_test_set.values[i][0])
    Y_test_as_array = np.array(Y_test_as_array)

    cv_scores = 100* np.mean(cross_val_score(classifier, X_train_set, Y_train_set, cv=k_folds, scoring='accuracy'))
    classifier_prediction = classifier.predict(X_test_set)
    f1 = f1_score(Y_test_set, classifier_prediction, average='micro') * 100
    accuracy = 100 * np.sum(Y_test_as_array == classifier_prediction) / len(Y_test_as_array)
    error = 100-accuracy
    precision = 100* precision_score(Y_test_set, classifier_prediction, average='micro')
    recall = 100* recall_score(Y_test, classifier_prediction, average='micro')

    print(classifier_name + " >>> CV score: " + str(cv_scores) + ", F1: " + str(f1) + ", Accuracy: " + str(accuracy) +
          ", Error: " + str(error) + ", Precision: " + str(precision) + ", Recall: " + str(recall))

k_folds = 10

# -------------------------------------------------------------------
# ---------------- 1: Load the prepared training set ----------------
# -------------------------------------------------------------------

X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file, header=None, names=['Vote'])


# -------------------------------------------------------------------
# ---------------- 2: Train generative model ------------------------
# -------------------------------------------------------------------


# QDA
QDA_classifier = QuadraticDiscriminantAnalysis()
QDA_classifier.fit(X_train, Y_train)

# LDA
LDA_classifier = LinearDiscriminantAnalysis()
LDA_classifier.fit(X_train, Y_train)

# Gaussian Naive Bayes
GNB_classifier = GaussianNB()
GNB_classifier.fit(X_train, Y_train)


# -------------------------------------------------------------------
# ---------------- 3: Train clustering model ------------------------
# -------------------------------------------------------------------

X_values = X_train.values
colors = (0, 0, 1)
area = np.pi

cols_num_wanted = 8

# for i in range(1, cols_num_wanted + 1):
#     for j in range(i, cols_num_wanted + 1):
#
#         # We don't want a comparison between a feature and itself
#         if i == j:
#             continue
#
#         kmeans = KMeans(n_clusters=2)
#         X_selected_values = X_values[:, [i, j]]
#         kmeans.fit(X_selected_values)
#
#         x = X_selected_values[:, 0]
#         y = X_selected_values[:, 1]
#
#         # Plot
#         plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#         plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], s=200, c='r', marker='s', label='center_1')
#         plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], s=200, c='g', marker='s', label='center_2')
#         plt.grid()
#         plt.legend()
#
#         plt.title(X_train.columns[i] + " vs. " + X_train.columns[j])
#         plt.xlabel(X_train.columns[i])
#         plt.ylabel(X_train.columns[j])
#         plt.show()


# ------------------------ 3A: Party-Cluster Belonging --------------


parties = ["Khakis", "Oranges", "Purples", "Turquoises", "Yellows", "Blues", "Whites",
           "Greens", "Violets", "Browns", "Reds", "Greys", "Pinks"]

X_train = X_train.set_index('Unnamed: 0')
all_data = pd.concat([X_train, Y_train], axis=1)

total = all_data.shape[0]

threshold = 0.985

parties_voting_percentage = {}

# Division according to Weighted_education_rank

cluster_A_parties_weighted = []
cluster_B_parties_weighted = []

print("Division according to Weighted_education_rank")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
    parties_voting_percentage[i] = all_data_copy.shape[0] / total  ## Filling the percentages of voting among the parties
    cluster_A_vals = all_data_copy[all_data_copy['Weighted_education_rank'] > 0]
    cluster_A_percent = cluster_A_vals.shape[0] / total
    cluster_B_percent = 1-cluster_A_percent
    print(i + ": Cluster A = " + str(cluster_A_percent) + ", Cluster B = " + str(cluster_B_percent))
    if cluster_B_percent > threshold:
        cluster_B_parties_weighted.append(i)
    else:
        cluster_A_parties_weighted.append(i)

print("**********")

# Division according to Avg_Residancy_Altitude

cluster_A_parties_residancy = []
cluster_B_parties_residancy = []

print("Division according to Avg_Residancy_Altitude")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
    cluster_A_vals = all_data_copy[all_data_copy['Avg_Residancy_Altitude'] > 0.25]
    cluster_A_percent = cluster_A_vals.shape[0] / total
    cluster_B_percent = 1-cluster_A_percent
    print(i + ": Cluster A = " + str(cluster_A_percent) + ", Cluster B = " + str(1-cluster_A_percent))
    if cluster_B_percent > threshold:
        cluster_B_parties_residancy.append(i)
    else:
        cluster_A_parties_residancy.append(i)

print("**********")


# Division according to Avg_education_importance

cluster_A_parties_education = []
cluster_B_parties_education = []

print("Division according to Avg_education_importance")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
    cluster_A_vals = all_data_copy[all_data_copy['Avg_education_importance'] > -0.5]
    cluster_A_percent = cluster_A_vals.shape[0] / total
    cluster_B_percent = 1-cluster_A_percent
    print(i + ": Cluster A = " + str(cluster_A_percent) + ", Cluster B = " + str(1-cluster_A_percent))
    if cluster_B_percent > threshold:
        cluster_B_parties_education.append(i)
    else:
        cluster_A_parties_education.append(i)

print("**********")


# ------------------------ 3B: Cluster-Voting percents --------------


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

# Avg_Residancy_Altitude

cluster_A_parties_residancy_percent = 0

for i in cluster_A_parties_residancy:
    cluster_A_parties_residancy_percent = cluster_A_parties_residancy_percent + parties_voting_percentage[i]

print("**********")

print("Cluster A percents (Residancy): " + str(cluster_A_parties_residancy_percent))
print(cluster_A_parties_residancy)
print("Cluster B percents (Residancy): " + str(1-cluster_A_parties_residancy_percent))
print(cluster_B_parties_residancy)


# Avg_education_importance

cluster_A_parties_education_percent = 0

for i in cluster_A_parties_education:
    cluster_A_parties_education_percent = cluster_A_parties_education_percent + parties_voting_percentage[i]

print("**********")

print("Cluster A percents (Education): " + str(cluster_A_parties_education_percent))
print(cluster_A_parties_education)
print("Cluster B percents (Education): " + str(1-cluster_A_parties_education_percent))
print(cluster_B_parties_education)


# -------------------------------------------------------------------
# ---------------- 4: Load the prepared test set --------------------
# -------------------------------------------------------------------

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)


Y_test_file = 'Y_test.csv'
Y_test = pd.read_csv(Y_test_file, header=None, names=['Vote'])

# -------------------------------------------------------------------
# ---------------- 4: Apply and check performance -------------------
# -------------------------------------------------------------------

printResults(QDA_classifier, X_test, Y_test, X_train, Y_train, "QDA")
printResults(LDA_classifier, X_test, Y_test, X_train, Y_train, "LDA")
printResults(GNB_classifier, X_test, Y_test, X_train, Y_train, "GNB")

LDA_probs = LDA_classifier.predict_proba(X_test)
GNB_probs = GNB_classifier.predict_proba(X_test)

top_two_parties_pairs = []

# TODO apply on GNB_probs as well

for i in LDA_probs:

    first_result = np.where(i == np.amax(i))
    first_index = first_result[0][0]

    np.delete(i, first_index)

    second_result = np.where(i == np.amax(i))
    second_index = second_result[0][0]

    if second_index == first_index:
        second_index = second_index + 1

    res_array = []
    res_array.append(first_index)
    res_array.append(second_index)

    top_two_parties_pairs.append(res_array)

# yakirs_function(top_two_parties_pairs)

# foreach example, take maximum
