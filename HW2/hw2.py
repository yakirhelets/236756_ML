import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from skrebate import ReliefF

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

print(elections_data.iloc[[1]].dtypes)

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

X_train_prep_mode_b1 = X_train_prep_mode.copy(deep=True)

# Multinominal/Polynominal
# iterate over all Multinominal/Polynominal columns
multiColumns = {'Most_Important_Issue', 'Will_vote_only_large_party', 'Age_group', 'Main_transportation', 'Occupation'}
for column in multiColumns:
    # for each column determine the set of unique values
    column_values_set = set(X_train_prep_mode_b1[column])

    for value in column_values_set:
        # split each column by the number of options to a binominal feature column
        value_dict = {value: 1}
        for val in column_values_set:
            if val != value:
                value_dict[val] = 0

        X_train_prep_mode_b1[column + '_' + str(value)] = pd.Series.copy(X_train_prep_mode_b1[column])
        X_train_prep_mode_b1[column + '_' + str(value)].replace(value_dict, inplace=True)

# delete original columns
X_train_prep_mode_b1.drop(columns=multiColumns, inplace=True)


# Binominal (converting to [-1,1])
# biColumns = {'Looking_at_poles_results', 'Married', 'Gender', 'Voting_Time', 'Financial_agenda_matters'}

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

X_train_prep_mode_b1.replace(cleanup_nums, inplace=True)



# ------------------------ B2: Outlier Detection --------------------

# Printing the correlation matrix TODO: research correlation in pandas more
plt.matshow(X_train_prep_mode_b1.corr())
plt.show()
# histogram_intersection = lambda a, b: np.minimum(a, b).sum().round(decimals=1)
# X_train_prep.corr(method=histogram_intersection)

# outlier detection - 3 examples of correlated features #TODO: provide more examples based on the correlation matrix
plt.scatter(X_train_prep_mode_b1.Avg_monthly_expense_when_under_age_21, X_train_prep_mode_b1.Avg_monthly_expense_on_pets_or_plants)
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

#TODO: bring following code back
# for i in range(len(X_train_prep.columns)):
#     i_th_column = X_train_prep.iloc[:, i]
#     hist = i_th_column.hist()
#     hist.set_title(X_train_prep.columns.values[i])
#     plt.show()

X_train_prep_mode_b1_scaled = X_train_prep_mode_b1.copy(deep=True)


# Z-score:

z_score_attributes = [1,2,3,5,14,17,18,19,20,21,22,24,25,26,27,28,29,30]

# actual zscore scaling
for i in z_score_attributes:
    X_train_prep_mode_b1_scaled.iloc[:, i] = stats.zscore(X_train_prep_mode_b1_scaled.iloc[:, i])


# Min-Max:

# TODO: (-1,1) range for all or change for some attributes?

scaler = MinMaxScaler(feature_range=(-1, 1))

# min_max attributes indices: [0,9,10,11,12,13,15,16,23]
min_max_attributes = ["Occupation_Satisfaction", "Financial_balance_score_(0-1)", "%Of_Household_Income",
                      "Yearly_IncomeK", "Overall_happiness_score", "Garden_sqr_meter_per_person_in_residancy_area",
                      "Yearly_ExpensesK", "%Time_invested_in_work", "%_satisfaction_financial_policy"]

# actual min_max scaling
X_train_prep_mode_b1_scaled[min_max_attributes] = scaler.fit_transform(X_train_prep_mode_b1_scaled[min_max_attributes])

X_train_prep_mode_b1_scaled.to_csv("updated_file.csv")


# -------------------------------------------------------------------
# ------------------------ D: Feature Selection ---------------------
# -------------------------------------------------------------------

# Wrapper method = SFS
# TODO: fill in
knn = KNeighborsClassifier(n_neighbors=3)  # TODO explain why chose this value
sfs = SFS(knn, k_features=20, forward=True, floating=False, verbose=2, scoring='accuracy', cv=0)  # TODO explain the values here
result_sfs = sfs.fit(X_train_prep_mode_b1_scaled, Y_train_prep)
print(result_sfs.subsets_)  # TODO make sure applying on the right X and Y and then right down results


# Filter method = Relief algorithm
# relief = ReliefF(n_features_to_select=20, n_neighbors=3)  # TODO explain the values here
# result_relief = relief.fit(X_train_prep_mode_b1_scaled, Y_train_prep)
# print(result_relief)  # TODO make sure applying on the right X and Y and then right down results




# TODO: lastly probably get the intersection of the two methods and choose only those features and explain in the report

# -------------------------------------------------------------------
# ------------------------ 5: Apply changes on other sets -----------
# -------------------------------------------------------------------

# TODO: fill in


# -------------------------------------------------------------------
# ------------------------ 6: Saving the prepared data --------------
# -------------------------------------------------------------------

# Save the 3x2 data sets in CSV files

# TODO: fill in
