import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor

def prepare_data():

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
    # print(elections_data.iloc[[1]].dtypes)

    # -------------------------------------------------------------------
    # ------------------------ 3: Split the data ------------------------
    # -------------------------------------------------------------------

    # Split the data to train, test, validation sets
    data = elections_data.iloc[:, 1:]  # data
    labels = elections_data.iloc[:, 0]  # labels

    X_train, X_test, Y_train, Y_test = \
        train_test_split(data, labels, train_size=0.8, test_size=0.2, shuffle=True,
                         random_state=None, stratify=None)
    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X_train, Y_train, train_size=0.8, test_size=0.2, shuffle=True,
                         random_state=None, stratify=None)


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
        # -----
        highest_corr_list = highest_corr_list[1::2]
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

    # Method 3 - Related Features

    def relatedFeaturesImputation(dataset):
        corr_list = getHighestCorrelations(dataset, -1.0)

        int_value_columns = {'Occupation_Satisfaction', 'Yearly_ExpensesK', 'Last_school_grades',
                             'Number_of_differnt_parties_voted_for',
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
                            df_first.iloc[j, 0] = actual_value

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
        multiColumns = {'Most_Important_Issue', 'Will_vote_only_large_party', 'Age_group', 'Main_transportation',
                        'Occupation'}
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
                to_remove = to_remove | (
                    set(dataset_X.loc[(dataset_X[col1] == X[i][0]) & (dataset_X[col2] <= X[i][1])].index.tolist()))
        to_remove = list(to_remove)
        dataset_X.drop(to_remove, inplace=True)
        dataset_Y.drop(to_remove, inplace=True)


    # Applying the multivariate outlier removal on all sets
    for tuple in highest_corr:
        # --- Actual removal ---
        RemoveMultiOutliers(X_train, Y_train, tuple[0], tuple[1])
        RemoveMultiOutliers(X_validation, Y_validation, tuple[0], tuple[1])
        RemoveMultiOutliers(X_test, Y_test, tuple[0], tuple[1])

    # -------------------------------------------------------------------
    # ------------------------ D: Normalization -------------------------
    # -------------------------------------------------------------------

    # First we examine the distribution of each column to see whether to apply Z-score on it or Min-Max:

    # Z-score:

    def zscoreScaling(dataset):
        z_score_attributes = [1, 2, 3, 5, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30]

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

    # Actual features that were published in hw3:
    selected_features = ['Avg_environmental_importance', 'Avg_government_satisfaction', 'Avg_education_importance',
                         'Most_Important_Issue_Financial', 'Most_Important_Issue_Other',
                         'Most_Important_Issue_Foreign_Affairs', 'Most_Important_Issue_Social',
                         'Most_Important_Issue_Healthcare', 'Most_Important_Issue_Education',
                         'Most_Important_Issue_Environment', 'Most_Important_Issue_Military',
                         'Avg_monthly_expense_on_pets_or_plants', 'Avg_Residancy_Altitude',
                         'Yearly_ExpensesK', 'Weighted_education_rank', 'Number_of_valued_Kneset_members']

    # Pick the selected features for each set
    X_train = X_train.filter(selected_features)
    X_validation = X_validation.filter(selected_features)
    X_test = X_test.filter(selected_features)


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
