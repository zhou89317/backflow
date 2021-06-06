from prepare_data import *
import datetime
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from featuremodel_parameterselection import show_featureimportance


def traintestsplit_standardization():
    """
    this function does train_test_split for time series and standardization
    :param finaldf: the df computed from function prepare_data(stocknameabrev, enddatetime, ndays):
    :return: X_train, X_test, y_train, y_test (X_train and X_test are all standardized version ndarray)
    """
    finaldf = prepare_data(STOCK_CODE,NDAYS_LOOKAHEAD)
    # note this excludes the index column
    X = finaldf.iloc[:,:-1]
    y = finaldf.iloc[:,-1]
    X_train, X_test, y_train, y_test = temporal_train_test_split(X,y,test_size=0.2)

    # use minmaxscalar instead of standardscalar since we do not know stock price distribution is normal or not
    scalar = MinMaxScaler()
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)
    #TODO: MAY BE CHANGE TO StandardScaler TO HAVE A TRY
    return X_train, X_test, y_train, y_test



def traintestsplit():
    """
        this function does train_test_split for time series but no standardization
        :param finaldf: the df computed from function prepare_data(stocknameabrev, enddatetime, ndays):
        :return: X_train, X_test, y_train, y_test (X_train and X_test are all unstandardized version Dataframe)
        """
    finaldf = prepare_data(STOCK_CODE,NDAYS_LOOKAHEAD)
    # note this excludes the index column
    X = finaldf.iloc[:, :-1]
    y = finaldf.iloc[:, -1]
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test



def preparedata_forSVM():
    """
    best model(for feature extraction) determined in file featuremodel_parameterselection.py to evaluate useful
    features and prune the columns of the dataset so that they can directly feed into out final SVM model

    :return: X_train,X_test,y_train,y_test (for final SVM model)
    """
    feature_importance = list(show_featureimportance())
    X_train, X_test, y_train, y_test = traintestsplit()
    sortedimportance_originalindex = sorted(enumerate(feature_importance), key=lambda x:x[1], reverse=True)

    i = 0
    total_feature_importance = 0
    for count,num in sortedimportance_originalindex:
        if total_feature_importance < 0.95:
            total_feature_importance += num
            i += 1
        else:
            break


    columns_iloc = []
    enumerate_chosen_features = list(sortedimportance_originalindex)[:i]
    for item in enumerate_chosen_features:
        columns_iloc.append(item[0])
    columns_iloc = sorted(columns_iloc)
    X_train = X_train.iloc[:, columns_iloc]
    X_test = X_test.iloc[:, columns_iloc]
    return X_train, X_test, y_train, y_test,columns_iloc



def apply_selectfeatures_realtime(X_train, X_test, y_train):
    """
    apply feature selection for data that is used for final prediction, only the last ndays data will be used for
    label prediction.
    :param X_train: training data for the final prediction, this is essentially all data except the last ndays ones
    :param X_test: this is just the last ndays data samples
    :param y_train: training labels for final prediction, this is essentially all labels except the last ndays labels
    :return:
    """
    feature_importance = list(show_featureimportance())
    sortedimportance_originalindex = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)

    i = 0
    total_feature_importance = 0
    for count, num in sortedimportance_originalindex:
        if total_feature_importance < 0.95:
            total_feature_importance += num
            i += 1
        else:
            break

    columns_iloc = []
    enumerate_chosen_features = list(sortedimportance_originalindex)[:i]
    for item in enumerate_chosen_features:
        columns_iloc.append(item[0])
    columns_iloc = sorted(columns_iloc)
    X_train = X_train.iloc[:, columns_iloc]
    X_test = X_test.iloc[:, columns_iloc]
    return X_train, X_test, y_train



