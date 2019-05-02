import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# -------------------------------------------------------------------
# ------------------------ 1: Loading the data ----------------------
# -------------------------------------------------------------------

# Load the Election Challenge data from the ElectionsData.csv
elections_file = 'ElectionsData.csv'
elections_data = pd.read_csv(elections_file)

# -------------------------------------------------------------------
# ------------------------ 2: Identify and set the type -------------
# -------------------------------------------------------------------

# Identify and set the correct type of each attribute
print(elections_data.iloc[[1]].dtypes)

# -------------------------------------------------------------------
# ------------------------ 3: Split the data ------------------------
# -------------------------------------------------------------------

# Split the data to train, test, validation sets
data = elections_data.iloc[:, 1:]  # data
labels = elections_data.iloc[:, 0]  # labels

X_train, X_test, Y_train, Y_test =\
    train_test_split(data, labels, train_size=0.8, test_size=0.2, shuffle=True,
                     random_state=None, stratify=None)
X_train, X_validation, Y_train, Y_validation =\
    train_test_split(X_train, Y_train, train_size=0.8, test_size=0.2, shuffle=True,
                     random_state=None ,stratify=None)

# -------------------------------------------------------------------
# ------------------------ 4: Data preparation actions --------------
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# ------------------------ Helper functions -------------------------
# -------------------------------------------------------------------

# Gives a highest correlation list of tuples of features
def getHighestCorrelations(dataset, dropping_value=0.8):
    # Printing the correlation matrix
    dataset.apply(lambda x: x.factorize()[0]).corr()
    plt.matshow(dataset.corr())
    plt.show()
    # Fetching the features with the highest correlation between them
    corr = dataset.corr().abs()
    s = corr.unstack()
    sorted_corrs = s.sort_values(kind="quicksort", ascending=False)
    highest_corr_list = []
    to_remove = []
    for idx in range(len(sorted_corrs)):
        if dropping_value <= sorted_corrs[idx] < 1.0:
            highest_corr_list.append(sorted_corrs.index[idx])
        # This part is used only for printing the highest correlations
        else:
            to_remove.append(sorted_corrs.index[idx])
        # -----
    sorted_corrs.drop(to_remove, inplace=True)
    # This part is used only for printing the highest correlations
    print("Highest correlations:")
    print(sorted_corrs.iloc[1::2])
    # -----
    highest_corr_list = highest_corr_list[1::2]
    corr.to_csv("corr_abs.csv")
    return highest_corr_list

# -------------------------------------------------------------------
# ------------------------ A: Data Cleanup --------------------------
# -------------------------------------------------------------------

# Data cleanup - remove negative values where it's invalid to have them

def cleanUpData(dataset):
    nonNegativeCols = {
        'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses', 'Avg_monthly_expense_on_pets_or_plants',
        'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Yearly_IncomeK',
        'Garden_sqr_meter_per_person_in_residancy_area', 'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
        '%Time_invested_in_work', 'Avg_monthly_household_cost', 'Phone_minutes_10_years', 'Avg_size_per_room',
        'Avg_monthly_income_all_years', 'Last_school_grades', 'Number_of_differnt_parties_voted_for',
        'Number_of_valued_Kneset_members', 'Num_of_kids_born_last_10_years'}

    for col in nonNegativeCols:
        i_th_column = dataset[[col]]
        i_th_column.where(i_th_column > 0, inplace=True, other=np.nan)
        dataset[[col]] = i_th_column

cleanUpData(X_train)
cleanUpData(X_validation)
cleanUpData(X_test)

# -------------------------------------------------------------------
# ------------------------ B: Imputation ----------------------------
# -------------------------------------------------------------------

# Method 1: Remove examples with at least one NaN

X_train_dropped_NaN = X_train.copy(deep=True)
X_train_dropped_NaN = X_train_dropped_NaN.dropna()
print("Remove examples with at least one NaN - INFO")
print(X_train_dropped_NaN.shape)

# Method 2: Single Value Assignment - Mode

X_train_mode = X_train.copy(deep=True)

highest_corr = getHighestCorrelations(X_train, dropping_value=0.8)
tuple = highest_corr[0]

# BEFORE IMPUTING MODE
plt.scatter(X_train_mode[tuple[0]], X_train_mode[tuple[1]])
plt.title("BEFORE IMPUTING MODE")
plt.xlabel(tuple[0])
plt.ylabel(tuple[1])
plt.show()

for i in range(len(X_train_mode.columns)):
    i_th_column = X_train_mode.iloc[:, i]
    mod = i_th_column.mode().get(0)
    X_train_mode.iloc[:, i].fillna(mod, inplace=True)

# AFTER IMPUTING MODE
plt.scatter(X_train_mode[tuple[0]], X_train_mode[tuple[1]])
plt.title("AFTER IMPUTING MODE")
plt.xlabel(tuple[0])
plt.ylabel(tuple[1])
plt.show()

# Method 3 - Related Features

def relatedFeaturesImputation(dataset):

    corr_list = getHighestCorrelations(dataset, -1.0)

    int_value_columns = {'Occupation_Satisfaction', 'Yearly_ExpensesK', 'Last_school_grades', 'Number_of_differnt_parties_voted_for',
                         'Number_of_valued_Kneset_members', 'Num_of_kids_born_last_10_years'}

    # Prepare a array that shows for each attribute the attribute that is closest to it
    # For each attribute:
    for i in range(len(dataset.columns)):
        # save the closest attribute to it
        lin_reg = LinearRegression()
        first_att_name = dataset.columns[i]
        try:
            found_tup = next(tup for tup in corr_list if (first_att_name in tup))
            second_att_name = found_tup[0] if first_att_name == found_tup[1] else found_tup[1]
            # drop all columns except these two
            df_two_cols = dataset[[first_att_name, second_att_name]]
            # save the examples that have both (don't have nan in any)
            df_have_both_values = df_two_cols.dropna()
            # make the linear line from all of these examples with the built in function
            df_first = df_two_cols[[first_att_name]]
            df_second = df_two_cols[[second_att_name]]
            lin_reg.fit(df_have_both_values[[second_att_name]], df_have_both_values[[first_att_name]])

            # Fill in the missing values
            for j in range(len(df_first)):
                if np.math.isnan(df_first.iloc[j, 0]):
                    if not np.math.isnan(df_second.iloc[j, 0]):
                        dat = {'col1': [df_second.iloc[j, 0]]}
                        predicted_value = lin_reg.predict(pd.DataFrame(dat))
                        actual_value = predicted_value[0][0]
                        if first_att_name in int_value_columns:
                            actual_value = round(actual_value)
                        df_first.iloc[j,0] = actual_value

            # assign the imputed column back to the original one
            dataset.iloc[:, i] = df_first.iloc[:, :]

        except StopIteration:
            continue


relatedFeaturesImputation(X_train)
relatedFeaturesImputation(X_validation)
relatedFeaturesImputation(X_test)

# Multinominal/Polynominal/Binominal imputation
# Fill missing values that could not be filled with linear regression - with mode()
def modeImputation(dataset):
    # for col in dataset.columns:
    #     i_th_column = dataset[[col]]
    #     mod = i_th_column.mode()
    #     dataset[[col]] = dataset[[col]].fillna(mod, inplace=True)

    for i in range(len(dataset.columns)):
        i_th_column = dataset.iloc[:, i]
        mod = i_th_column.mode().get(0)
        dataset.iloc[:, i].fillna(mod, inplace=True)

modeImputation(X_train)
modeImputation(X_validation)
modeImputation(X_test)


# -------------------------------------------------------------------
# ------------------------ C: Data Cleansing ------------------------
# -------------------------------------------------------------------

# ------------------------ C1: Type/Value Modification --------------

# Multinominal/Polynominal
# iterate over all Multinominal/Polynominal columns
def multinominalModification(dataset):
    multiColumns = {'Most_Important_Issue', 'Will_vote_only_large_party', 'Age_group', 'Main_transportation', 'Occupation'}
    for column in multiColumns:
        # for each column determine the set of unique values
        column_values_set = set(dataset[column])

        for value in column_values_set:
            # split each column by the number of options to a binominal feature column
            value_dict = {value: 1}
            for val in column_values_set:
                if val != value:
                    value_dict[val] = 0

            dataset[column + '_' + str(value)] = pd.Series.copy(dataset[column])
            dataset[column + '_' + str(value)].replace(value_dict, inplace=True)

    # delete original columns
    dataset.drop(columns=multiColumns, inplace=True)


multinominalModification(X_train)
multinominalModification(X_validation)
multinominalModification(X_test)

# Binominal (converting to [-1,1])

def binominalModification(dataset):
    # We define an 'encoding' dictionary from a binominal feature to a numerical value
    cleanup_nums = {"Financial_agenda_matters":
                        {"Yes": 1, "No": -1},
                    "Married":
                        {"Yes": 1, "No": -1},
                    "Looking_at_poles_results":
                        {"Yes": 1, "No": -1},
                    "Gender":
                        {"Male": 1, "Female": -1},
                    "Voting_Time":
                        {"By_16:00": 1, "After_16:00": -1}
                    }

    dataset.replace(cleanup_nums, inplace=True)

binominalModification(X_train)
binominalModification(X_validation)
binominalModification(X_test)

# ------------------------ C2: Outlier Detection --------------------

highest_corr = getHighestCorrelations(X_train, dropping_value=0.8)

# Univariate Outliers - One dimensional (one variable)
# Methods of outlier detection:
# 1. Any value, which is beyond the range of -1.5 x IQR to 1.5 x IQR
# 2. Use capping methods. Any value which out of range of 5th and 95th percentile can be considered as outlier
# 3. Data points, three or more standard deviation away from mean are considered outlier (Z-Score)
def clipByIQR(dataset_X, dataset_Y, dropping_factor=1.5):
    to_remove = set()
    for col in X_train.columns.tolist():
        # # Printing
        # plt.hist(X_train_prep_mode_b1[col])
        # plt.show()
        sort = sorted(dataset_X[col])
        q1, q3 = np.percentile(sort, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (dropping_factor * iqr)
        upper_bound = q3 + (dropping_factor * iqr)
        for row in dataset_X.iterrows():
            if row[1][col] < lower_bound or row[1][col] > upper_bound:
                to_remove.add(row[0])
        # # Printing
        # plt.hist(X_train[col])
        # plt.show()

    to_remove = list(to_remove)
    dataset_X.drop(to_remove, inplace=True)
    dataset_Y.drop(to_remove, inplace=True)


def clipByPerecentile(dataset_X, dataset_Y, dropping_percentage=5):
    to_remove = set()
    for col in X_train.columns.tolist():
        # # Printing
        # plt.hist(X_train[col])
        # plt.show()
        sort = sorted(dataset_X[col])
        lower_bound, upper_bound = np.percentile(sort, [dropping_percentage, 100 - dropping_percentage])
        for row in dataset_X.iterrows():
            if row[1][col] < lower_bound or row[1][col] > upper_bound:
                to_remove.add(row[0])
        # # Printing
        # plt.hist(X_train[col])
        # plt.show()
    to_remove = list(to_remove)
    dataset_X.drop(to_remove, inplace=True)
    dataset_Y.drop(to_remove, inplace=True)


def clipByZScore(dataset_X, dataset_Y, dropping_factor=3.0):
    to_remove = set()
    for col in X_train.columns.tolist():
        # # Printing
        # plt.hist(X_train[col])
        # plt.show()
        z_scores = stats.zscore(dataset_X[col])
        for idx in range(len(z_scores)):
            if abs(z_scores[idx]) > dropping_factor:
                to_remove.add(idx)
        # # Printing
        # plt.hist(X_train[col])
        # plt.show()
    to_remove = list(to_remove)
    dataset_X.drop(to_remove, inplace=True)
    dataset_Y.drop(to_remove, inplace=True)


# Multivariate Outliers (more than one variable)
# Using Local Outlier Factor in order to detect multivariate outliers
def RemoveMultiOutliers(dataset_X, dataset_Y, col1, col2):
    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    X = list(zip(dataset_X[col1].values.tolist(),
                 dataset_X[col2].values.tolist()))
    y_pred = clf._fit_predict(X)
    to_remove = set()
    for i in range(len(X)):
        if y_pred[i] == -1:
            to_remove = to_remove | (set(dataset_X.loc[(dataset_X[col1] == X[i][0]) & (dataset_X[col2] <= X[i][1])].index.tolist()))
    to_remove = list(to_remove)
    dataset_X.drop(to_remove, inplace=True)
    dataset_Y.drop(to_remove, inplace=True)


# Applying the multivariate outlier removal on all sets
for tuple in highest_corr:
    plt.subplot(121).scatter(X_train[tuple[0]], X_train[tuple[1]])
    # Plot - before (train only)
    plt.title("Before outlier removal")
    plt.xlabel(tuple[0])
    plt.ylabel(tuple[1])

    # --- Actual removal ---
    RemoveMultiOutliers(X_train, Y_train, tuple[0], tuple[1])
    RemoveMultiOutliers(X_validation, Y_validation, tuple[0], tuple[1])
    RemoveMultiOutliers(X_test, Y_test, tuple[0], tuple[1])

    # Plot - after (train only)
    plt.subplot(122).scatter(X_train[tuple[0]], X_train[tuple[1]])
    plt.title("After outlier removal")
    plt.xlabel(tuple[0])
    plt.ylabel(tuple[1])
    plt.show()
    # ---


# -------------------------------------------------------------------
# ------------------------ D: Normalization -------------------------
# -------------------------------------------------------------------

# First we examine the distribution of each column to see whether to apply Z-score on it or Min-Max:

# plot the histogram for each column of X_train_prep and check if it's normally or uniformly distributed

for i in range(len(X_train.columns)):
    i_th_column = X_train.iloc[:, i]
    hist = i_th_column.hist()
    hist.set_title(X_train.columns.values[i])
    plt.show()

# Z-score:

def zscoreScaling(dataset):
    z_score_attributes = [1,2,3,5,14,17,18,19,20,21,22,24,25,26,27,28,29,30]

    # actual zscore scaling
    for i in z_score_attributes:
        dataset.iloc[:, i] = stats.zscore(dataset.iloc[:, i])


zscoreScaling(X_train)
zscoreScaling(X_validation)
zscoreScaling(X_test)

# Min-Max:

def minmaxScaling(dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # min_max attributes indices: [0,9,10,11,12,13,15,16,23]
    min_max_attributes = ["Occupation_Satisfaction", "Financial_balance_score_(0-1)", "%Of_Household_Income",
                          "Yearly_IncomeK", "Overall_happiness_score", "Garden_sqr_meter_per_person_in_residancy_area",
                          "Yearly_ExpensesK", "%Time_invested_in_work", "%_satisfaction_financial_policy"]

    # actual min_max scaling
    dataset[min_max_attributes] = scaler.fit_transform(dataset[min_max_attributes])


minmaxScaling(X_train)
minmaxScaling(X_validation)
minmaxScaling(X_test)

# -------------------------------------------------------------------
# ------------------------ E: Feature Selection ---------------------
# -------------------------------------------------------------------
# Being done only on the train set to determine the features to select

# Filter method = Variance Threshold
filter = VarianceThreshold(threshold=0.2)
filter.fit_transform(X_train)
print(filter.get_support(indices=True))


# Wrapper method = SFS
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SFS(knn, k_features=30, forward=True, floating=False, verbose=2, scoring='accuracy', cv=0)
sfs = sfs.fit(X_train, Y_train)
print(sfs.k_feature_idx_)


# -------------------------------------------------------------------
# ------------------------ 5: Saving the prepared data --------------
# -------------------------------------------------------------------

# Save the 3x2 data sets in CSV files

X_train.to_csv("X_train.csv")
X_validation.to_csv("X_validation.csv")
X_test.to_csv("X_test.csv")

Y_train.to_csv("Y_train.csv")
Y_validation.to_csv("Y_validation.csv")
Y_test.to_csv("Y_test.csv")
