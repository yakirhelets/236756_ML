import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------
# ------------------------ 1: Loading the data ----------------------
# -------------------------------------------------------------------

# Load the Election Challenge data from the ElectionsData.csv
elections_file = 'ElectionsData.csv'
elections_data = pd.read_csv(elections_file)
# print(elections_data)

# -------------------------------------------------------------------
# ------------------------ 2: Identify and set the type -------------
# -------------------------------------------------------------------

# Identify and set the correct type of each attribute


# lst = list(data_frame._get_axis(1))
# print(lst)
# data_frame.astype()
# print(data_frame.sample(1).dtypes)

# TODO: make sure the datatypes are also int etc.
# TODO: probably after imputation, if not - ask
# print(elections_data.iloc[[1]].dtypes)



# -------------------------------------------------------------------
# ------------------------ 3: Split the data ------------------------
# -------------------------------------------------------------------

# Split the data to train, test, validation sets
data = elections_data.iloc[:, 1:]  # data
labels = elections_data.iloc[:, 0]  # labels
X_train, X_test, Y_train, Y_test =\
    train_test_split(data, labels, train_size=0.9, test_size=0.1, shuffle=False,
                     random_state=None, stratify=None) #TODO: changes train size and shuffle
X_train, X_validation, Y_train, Y_validation =\
    train_test_split(X_train, Y_train, train_size=0.83, test_size=0.17, shuffle=False,
                     random_state=None ,stratify=None) #TODO: changes train size and shuffle

X_train_prep = X_train.copy()
X_validation_prep = X_validation.copy()
Y_train_prep = Y_train.copy()
Y_validation_prep = Y_validation.copy()
X_test_prep = X_test.copy()
Y_test_prep = Y_test.copy()

# Data preparation actions on the training set:

# -------------------------------------------------------------------
# ------------------------ 4: Data preparation actions --------------
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# ------------------------ A: Imputation ----------------------------
# -------------------------------------------------------------------

# Method 1: Remove examples with at least one NaN

X_train_dropped_NaN = X_train_prep.dropna()
print(X_train_dropped_NaN.shape)

# Method 2: Single Value Assignment - Mode

X_train_prep_mode = X_train_prep.copy(deep=True)

for i in range(len(X_train_prep_mode.columns)):
    i_th_column = X_train_prep_mode.iloc[:, i]
    mod = i_th_column.mode().get(0)
    X_train_prep_mode.iloc[:, i] = X_train_prep_mode.iloc[:, i].fillna(mod)

print(X_train_prep_mode)

# Method 3:

# TODO: fill in


# -------------------------------------------------------------------
# ------------------------ B: Data Cleansing ------------------------
# -------------------------------------------------------------------

# ------------------------ B1: Type/Value Modification --------------

# Multinominal/Polynominal
# iterate over all Multinominal/Polynominal columns
multiColumns = {'Most_Important_Issue', 'Will_vote_only_large_party', 'Age_group', 'Main_transportation', 'Occupation'}
for column in multiColumns:
    # for each column determine the set of unique values
    column_values_set = set(X_train_prep[column])

    for value in column_values_set:
        # split each column by the number of options to a binominal feature column
        value_dict = {value: 1}
        for val in column_values_set:
            if val != value:
                value_dict[val] = 0

        X_train_prep[column + '_' + str(value)] = pd.Series.copy(X_train_prep[column])
        X_train_prep[column + '_' + str(value)].replace(value_dict, inplace=True)

# delete original columns
X_train_prep.drop(columns=multiColumns, inplace=True)


# Binominal (converting to [-1,1])
# biColumns = {'Looking_at_poles_results', 'Married', 'Gender', 'Voting_Time', 'Financial_agenda_matters'}

# We define an 'encoding' dictionary from a binominal feature to a numerical value
cleanup_nums = {"Financial_agenda_matters":
                    {"Yes": 1, "No": -1},
                "Married":
                    {"Yes": 1, "No": -1},
                "Gender":
                    {"Male": 1, "Female": -1},
                "Voting_Time":
                    {"By_16:00": 1, "After_16:00": -1}
                }

X_train_prep.replace(cleanup_nums, inplace=True)


# print(X_train_prep.values[:, 2])

# ------------------------ B2: Outlier Detection --------------------

# Printing the correlation matrix TODO: research correlation in pandas more
# TODO: right now shows only the 27 numerical features, need to convert caterogical to numerical before running
plt.matshow(X_train_prep.corr())
plt.show()
# histogram_intersection = lambda a, b: np.minimum(a, b).sum().round(decimals=1)
# X_train_prep.corr(method=histogram_intersection)

# outlier detection - 3 examples of correlated features #TODO: provide more examples based on the correlation matrix
plt.scatter(X_train_prep.Avg_monthly_expense_when_under_age_21, X_train_prep.Avg_monthly_expense_on_pets_or_plants)
plt.show()

# TODO: write code that eliminates examples beyond a certain range, for each attribute

# outliers_estimator = Estimator()
# outliers_detection = estimator.fit(X_train_prep)
# print(outliers_detection)


# -------------------------------------------------------------------
# ------------------------ C: Normalization -------------------------
# -------------------------------------------------------------------

# First we check the distribution of each column to see whether to apply
# Z-score on it or Min-Max:

# plot the histogram for each column of X_train_prep and check if it's normally or uniformly distributed

for i in range(len(X_train_prep_mode.columns)):
    i_th_column = X_train_prep.iloc[:, i]
    hist = i_th_column.hist()
    hist.set_title(X_train_prep.columns.values[i])
    plt.show()



# Z-score:

X_train_prep[0] = stats.zscore(X_train_prep[0]) #TODO continue to work on

# Min-Max:

# TODO: determine later for each attribute that we are going
# TODO: to apply min-max on if [-1,1] is right (leave as is) or we need a different range
# TODO: then create a new scaler with the required range

# scaler = MinMaxScaler(feature_range=(-1,1))

# column_min_max = X_train_prep["Avg_monthly_expense_when_under_age_21"]
# X_train_prep["Avg_monthly_expense_when_under_age_21"] = scaler.transform(column_min_max)



# -------------------------------------------------------------------
# ------------------------ D: Feature Selection ---------------------
# -------------------------------------------------------------------

# TODO: fill in


# -------------------------------------------------------------------
# ------------------------ 5: Apply changes on other sets -----------
# -------------------------------------------------------------------

# TODO: fill in


# -------------------------------------------------------------------
# ------------------------ 6: Saving the prepared data --------------
# -------------------------------------------------------------------

# Save the 3x2 data sets in CSV files

# TODO: fill in
