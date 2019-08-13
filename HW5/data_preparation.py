import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def prepare_data():

    # -------------------------------------------------------------------
    # ------------------------ 1: Loading the data ----------------------
    # -------------------------------------------------------------------

    np.random.seed(seed=0)

    # Load the Election Challenge data from the ElectionsData.csv
    elections_file = 'ElectionsData.csv'
    elections_data = pd.read_csv(elections_file)

    new_elections_file = 'ElectionsData_Pred_Features.csv'
    new_elections_data = pd.read_csv(new_elections_file)

    # -------------------------------------------------------------------
    # ------------------------ 2: Feature Selection ---------------------
    # -------------------------------------------------------------------

    # Actual features that were published in hw3:
    selected_features = ['Vote', 'Avg_environmental_importance', 'Avg_government_satisfaction', 'Avg_education_importance',
                         'Most_Important_Issue', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_Residancy_Altitude',
                         'Yearly_ExpensesK', 'Weighted_education_rank', 'Number_of_valued_Kneset_members']

    new_data_selected_features = ['IdentityCard_Num', 'Avg_environmental_importance', 'Avg_government_satisfaction', 'Avg_education_importance',
                         'Most_Important_Issue', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_Residancy_Altitude',
                         'Yearly_ExpensesK', 'Weighted_education_rank', 'Number_of_valued_Kneset_members']

    # Pick the selected features for each set
    elections_data = elections_data.filter(selected_features)
    new_elections_data = new_elections_data.filter(new_data_selected_features)


    # Correlations with right features - for Bonus part
    elections_data_copy = elections_data.copy(deep=True)
    for i in selected_features:
        if i == 'Vote' or i == 'Most_Important_Issue':
            continue
        corr = elections_data_copy.Vote.str.get_dummies().corrwith(elections_data_copy[i]/elections_data_copy[i].max())
        # print(corr)

    # -------------------------------------------------------------------
    # ------------------------ 3: Split the data ------------------------
    # -------------------------------------------------------------------

    # Split the data to train, test, validation sets
    data = elections_data.iloc[:, 1:]  # data
    labels = elections_data.iloc[:, 0]  # labels

    X_train, X_test, Y_train, Y_test = \
        train_test_split(data, labels, train_size=0.7, test_size=0.3, shuffle=True,
                         random_state=None, stratify=None)
    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X_train, Y_train, train_size=0.8, test_size=0.2, shuffle=True,
                         random_state=None, stratify=None)


    # -------------------------------------------------------------------
    # ------------------------ 4: Data preparation actions --------------
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # ------------------------ A: Data Cleanup --------------------------
    # -------------------------------------------------------------------

    # Data cleanup - remove negative values where it's invalid to have them

    def cleanUpData(dataset):
        nonNegativeCols = {
            'Avg_monthly_expense_on_pets_or_plants',
            'Avg_Residancy_Altitude',
            'Yearly_ExpensesK',
            'Number_of_valued_Kneset_members'}

        for col in nonNegativeCols:
            i_th_column = dataset[[col]]
            i_th_column.where(i_th_column > 0, inplace=True, other=np.nan)
            dataset[[col]] = i_th_column


    cleanUpData(X_train)
    cleanUpData(X_validation)
    cleanUpData(X_test)

    cleanUpData(new_elections_data)

    # -------------------------------------------------------------------
    # ------------------------ B: Imputation ----------------------------
    # -------------------------------------------------------------------

    # Multinominal/Polynominal/Binominal imputation
    # Fill missing values that could not be filled with linear regression - with mode()
    def modeImputation(dataset):

        for i in range(len(dataset.columns)):
            i_th_column = dataset.iloc[:, i]
            mod = i_th_column.mode().get(0)
            dataset.iloc[:, i].fillna(mod, inplace=True)


    modeImputation(X_train)
    modeImputation(X_validation)
    modeImputation(X_test)

    # imputation for new data set (hw5)

    for i in range(1, len(new_elections_data.columns)):
        i_th_column = new_elections_data.iloc[:, i]
        mod = i_th_column.mode().get(0)
        new_elections_data.iloc[:, i].fillna(mod, inplace=True)



    # -------------------------------------------------------------------
    # ------------------------ C: Data Cleansing ------------------------
    # -------------------------------------------------------------------

    # ------------------------ C1: Type/Value Modification --------------

    # Multinominal/Polynominal
    # iterate over all Multinominal/Polynominal columns
    def multinominalModification(dataset):
        multiColumns = {'Most_Important_Issue'}
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

    multinominalModification(new_elections_data)

    # -------------------------------------------------------------------
    # ------------------------ D: Normalization -------------------------
    # -------------------------------------------------------------------

    # First we examine the distribution of each column to see whether to apply Z-score on it or Min-Max:

    # Z-score:

    def zscoreScaling(dataset):
        # z_score_attributes = []
        z_score_attributes = [5]

        # actual zscore scaling
        for i in z_score_attributes:
            dataset.iloc[:, i] = stats.zscore(dataset.iloc[:, i])


    zscoreScaling(X_train)
    zscoreScaling(X_validation)
    zscoreScaling(X_test)


    # z_score_attributes = []
    z_score_attributes = [6]

    # actual zscore scaling
    for i in z_score_attributes:
        new_elections_data.iloc[:, i] = stats.zscore(new_elections_data.iloc[:, i])


    def minmaxScaling(dataset):
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # min_max_attributes = ["Weighted_education_rank", "Number_of_valued_Kneset_members",
        #                       "Avg_Residancy_Altitude", "Avg_environmental_importance",
        #                       "Avg_government_satisfaction", "Avg_education_importance",
        #                       "Avg_monthly_expense_on_pets_or_plants", "Yearly_ExpensesK"]
        min_max_attributes = ["Weighted_education_rank", "Number_of_valued_Kneset_members",
                              "Avg_Residancy_Altitude", "Avg_environmental_importance",
                              "Avg_government_satisfaction", "Avg_education_importance",
                              "Avg_monthly_expense_on_pets_or_plants"]

        # actual min_max scaling
        dataset[min_max_attributes] = scaler.fit_transform(dataset[min_max_attributes])

    minmaxScaling(X_train)
    minmaxScaling(X_validation)
    minmaxScaling(X_test)

    minmaxScaling(new_elections_data)


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

    new_elections_data.to_csv("X_new_data.csv")

