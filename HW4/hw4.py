import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from HW3 import data_preparation, automatic_model_selection
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# -------------------------------------------------------------------
# ------------------------ *: Helper functions ----------------------
# -------------------------------------------------------------------


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

# TODO: use cross validation



# -------------------------------------------------------------------
# ---------------- 3: Train clustering model ------------------------
# -------------------------------------------------------------------

# TODO: use cross validation

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


# TODO for each voting class, check how many percent have weighted_edu_rank > 0 and how many percent have < 0, print to console.

parties = ["Khakis", "Oranges", "Purples", "Turquoises", "Yellows", "Blues", "Whites",
           "Greens", "Violets", "Browns", "Reds", "Greys", "Pinks"]

X_train = X_train.set_index('Unnamed: 0')
all_data = pd.concat([X_train, Y_train], axis=1)

total = all_data.shape[0]

threshold = 0.97

# Division according to Weighted_education_rank

cluster_A_parties_weighted = []
cluster_B_parties_weighted = []

print("Division according to Weighted_education_rank")
for i in parties:
    all_data_copy = all_data.copy(deep=True)
    all_data_copy = all_data_copy[all_data_copy['Vote'] == i]
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


# ------------------------ 3B: Cluster-Voting percents --------------






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





# cross_val_scores = cross_val_score(GM_classifier, X_train, Y_train, cv=k_folds, scoring='accuracy')
# print(cross_val_scores)
