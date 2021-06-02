from prepare_data import *
import datetime

def prepare_data(stocknameabrev, enddatetime, ndays):
    """
        API KEY(Quandl): Jaq5cjYcrvukouwWEqwA
        perform everything in module prepare_data
        :param: stocknameabrev: a string represents a stock in American stock exchanges e.g 'AAPL.US'
        :param: enddatetime: a datetime instance represents the last trading day we want to fetch data
        :return: a dataframe ready for feature selection.
    """
    df, df_nasdaq, df_sp500 = readdata(stocknameabrev, enddatetime)
    df_poped, df_nasdaq_poped, df_sp500_poped = popfeatures(df, df_nasdaq, df_sp500, ndays=ndays)
    concatenated = integratedframes(df_poped, df_nasdaq_poped, df_sp500_poped)
    finaldf = labelling(concatenated, ndays=ndays)
    return finaldf