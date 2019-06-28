from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Returns the best model with the best parameters

k_folds = 10

def k_fold_cv_score(KNN_classifier, X_train, Y_train):
    # KFCV
    cross_val_scores = cross_val_score(KNN_classifier, X_train, Y_train, cv=k_folds, scoring='accuracy')
    # avergae score of classifier
    return cross_val_scores.mean()

def select_automatically(X_train, Y_train):
    result_classifier = ""
    score = 0

    for i in (1, 3, 5, 7, 9, 13, 15):
        KNN_classifier = KNeighborsClassifier(n_neighbors=i)
        clf_score = k_fold_cv_score(KNN_classifier, X_train, Y_train)
        if clf_score > score:
            score = clf_score
            result_classifier = "Selected = KNN: " + "K = " + str(i) + ", with score: " + str(score)

    for cr in ("gini", "entropy"):
        for min_sam in (1.0, 2, 3):
            DT_classifier = DecisionTreeClassifier(criterion=cr, min_samples_split=min_sam)
            clf_score = k_fold_cv_score(DT_classifier, X_train, Y_train)
            if clf_score > score:
                score = clf_score
                result_classifier = "Selected = Decision Tree: " + "criterion = " + str(cr) + ", " + "min_samples_split = " + str(min_sam) + ", with score: " + str(score)

    for n_est in (3, 10, 13):
        for max_d in (3, 5, None):
            RF_classifier = RandomForestClassifier(n_estimators=n_est, max_depth=max_d)
            clf_score = k_fold_cv_score(RF_classifier, X_train, Y_train)
            if clf_score > score:
                score = clf_score
                result_classifier = "Selected = Random Forest: " + "n_estimators = " + str(n_est) + ", " + "max_depth = " + str(max_d) + ", with score: " + str(score)

    for max_it in (10, 50, 100, 200, 500, 1000, 2000):
        Per_classifier = Perceptron(max_iter=max_it)
        clf_score = k_fold_cv_score(Per_classifier, X_train, Y_train)
        if clf_score > score:
            score = clf_score
            result_classifier = "Selected = Perceptron: " + "max_iter = " + str(max_it) + ", with score: " + str(score)

    for var_sm in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        GNB_classifier = GaussianNB(priors=None, var_smoothing=var_sm)
        clf_score = k_fold_cv_score(GNB_classifier, X_train, Y_train)
        if clf_score > score:
            score = clf_score
            result_classifier = "Selected = GNB_classifier: " + "var_smoothing = " + str(var_sm) + ", with score: " + str(score)

    return result_classifier
