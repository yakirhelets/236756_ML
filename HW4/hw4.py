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

#TODO: ask if we can use code form the tutorial

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width = 2 * np.sqrt(s)
        height = 2 * np.sqrt(s)
    else:
        angle = 0
        width = 2 * np.sqrt(covariance)
        height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


k_folds = 10

# -------------------------------------------------------------------
# ---------------- 1: Load the prepared training set ----------------
# -------------------------------------------------------------------

X_train_file = 'X_train.csv'
X_train = pd.read_csv(X_train_file)

Y_train_file = 'Y_train.csv'
Y_train = pd.read_csv(Y_train_file, header=None, names=['Vote'])


# -------------------------------------------------------------------
# ---------------- 2: Train generative and clustering models --------
# -------------------------------------------------------------------

# TODO: use cross validation

GM_classifier = GaussianMixture(n_components=1, max_iter=100)

# -------------------------------------------------------------------
# ---------------- 3: Load the prepared test set --------------------
# -------------------------------------------------------------------

X_test_file = 'X_test.csv'
X_test = pd.read_csv(X_test_file)


Y_test_file = 'Y_test.csv'
Y_test = pd.read_csv(Y_test_file, header=None, names=['Vote'])

# -------------------------------------------------------------------
# ---------------- 4: Apply and check performance -------------------
# -------------------------------------------------------------------

# Generative model

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(1,1,1)
# ax.grid()
# plot_gmm(GM_classifier, X_train.values, ax=ax)




# cross_val_scores = cross_val_score(GM_classifier, X_train, Y_train, cv=k_folds, scoring='accuracy')
# print(cross_val_scores)


# Clustering model

X_values = X_train.values
colors = (0, 0, 1)
area = np.pi

cols_num_wanted = 8

for i in range(1, cols_num_wanted + 1):
    for j in range(i, cols_num_wanted + 1):

        if i == j:
            continue
        kmeans = KMeans(n_clusters=2)  # 2 because we want a coalition and an opposition - #TODO add to report
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

        plt.title(X_train.columns[i] + " vs. " + X_train.columns[j])
        plt.xlabel(X_train.columns[i])
        plt.ylabel(X_train.columns[j])
        plt.show()
