import datetime
import pandas_datareader.data as web
import pandas as pd
import talib as tb
import os

STOCK_CODE = 'NVDA.US'
NDAYS_LOOKAHEAD = 5

def getlastweekdate():
    """
    get the last week day
    :return: a datetime instance representing the last weekday
    """
    now = datetime.date.today()
    result_date = now
    if now.isoweekday() == 1:
        result_date = now - datetime.timedelta(3)
    elif now.isoweekday() == 7:
        result_date = now - datetime.timedelta(2)
    else:
        result_date = now - datetime.timedelta(1)

    return result_date



def readdata(stocknameabrev):
    """
    API KEY(Quandl): Jaq5cjYcrvukouwWEqwA
    :param: stocknameabrev: a string represents a stock in American stock exchanges e.g 'AAPL.US'
    :param: enddatetime: a datetime instance represents the last trading day we want to fetch data
    :returns: a Dataframe of the selected stock
              a Dataframe of nasdap index
              a Dataframe of sp500 index
              Note that the three Dataframes returned are in the same time range.
    """
    start = datetime.datetime(2005, 1, 1)
    end = getlastweekdate()
    if os.path.exists('df.csv') and os.path.exists('df_sp500.csv') and os.path.exists('df_nasdaq.csv'):

        df = pd.read_csv('df.csv', index_col='Date',parse_dates=['Date'])
        df_sp500 = pd.read_csv('df_sp500.csv',index_col='Date',parse_dates=['Date'])
        df_nasdaq = pd.read_csv('df_nasdaq.csv',index_col='Date',parse_dates=['Date'])
    else:
        df = web.DataReader(stocknameabrev, 'stooq',start = start, end = end)
        df_sp500 = web.DataReader('^SPX', 'stooq', start = start, end = end)
        df_nasdaq = web.DataReader('^NDQ', 'stooq', start = start, end = end)
        df.fillna(method='bfill',inplace=True)
        list1 = list(df.index)
        list2 = list(df_nasdaq.index)
        if len(list2) > len(list1):
            for date in list1:
                list2.remove(date)

        df_nasdaq.drop(list2, inplace=True)
        df = df.iloc[::-1]
        df_sp500 = df_sp500.iloc[::-1]
        df_nasdaq = df_nasdaq.iloc[::-1]
        print(df.info)
        df.to_csv('df.csv')
        df_sp500.to_csv('df_sp500.csv')
        df_nasdaq.to_csv('df_nasdaq.csv')

    return df, df_nasdaq, df_sp500


def popfeatures(df, df_nasdaq, df_sp500, ndays=NDAYS_LOOKAHEAD):
    """
    :param df: df returned from function def readdata(stocknameabrev):
    :param df_nasdaq: df_nasdaq returned from function def readdata(stocknameabrev)
    :param df_sp500: df_sp500 returned from function def readdata(stocknameabrev)
    :param ndays: an integer representing how many days look-ahead we want to predict,
    must be consistent with ndays in function labelling
    :returns: df_fullindicators, df_nasdaq_fullindicators, df_sp500_fullindicators, three dataframes each with all
    technical indicators fully added.
    """
    # add WILLR - Williams' %R for past 10 trading days
    WILLR_numpy_df = tb.WILLR(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=14)
    df['WILLR'] = WILLR_numpy_df
    WILLR_numpy_dfnasdaq = tb.WILLR(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                    df_nasdaq['Close'].to_numpy(), timeperiod=14)
    df_nasdaq['WILLR_N'] = WILLR_numpy_dfnasdaq
    WILLR_numpy_dfSP500 = tb.WILLR(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                                   df_sp500['Close'].to_numpy(), timeperiod=14)
    df_sp500['WILLR_S'] = WILLR_numpy_dfSP500

    # add RSI14 - relative strength index for the past 14 days
    RSI14_numpy_df = tb.RSI(df['Close'].to_numpy(), timeperiod=14)
    df['RSI14'] = RSI14_numpy_df
    RSI14_numpy_dfnasdaq = tb.RSI(df_nasdaq['Close'].to_numpy(), timeperiod=14)
    df_nasdaq['RSI14_N'] = RSI14_numpy_dfnasdaq
    RSI14_numpy_dfSP500 = tb.RSI(df_sp500['Close'].to_numpy(), timeperiod=14)
    df_sp500['RSI14_S'] = RSI14_numpy_dfSP500

    # add CCI - commodity channel index for the past 12 days
    CCI12_numpy_df = tb.CCI(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=12)
    df['CCI12'] = CCI12_numpy_df
    CCI12_numpy_dfnasdaq = tb.CCI(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                    df_nasdaq['Close'].to_numpy(), timeperiod=12)
    df_nasdaq['CCI12_N'] = CCI12_numpy_dfnasdaq
    CCI12_numpy_dfSP500 = tb.CCI(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                                 df_sp500['Close'].to_numpy(), timeperiod=12)
    df_sp500['CCI12_S'] = CCI12_numpy_dfSP500

    # add MACD
    macd, macdsignal, macdhist = tb.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    macd_na, macdsignal_na, macdhist_na = tb.MACD(df_nasdaq['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_nasdaq['macd_N'] = macd_na
    df_nasdaq['macdsignal_N'] = macdsignal_na
    df_nasdaq['macdhist_N'] = macdhist_na
    macd_sp, macdsignal_sp, macdhist_sp = tb.MACD(df_sp500['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_sp500['macd_S'] = macd_sp
    df_sp500['macdsignal_S'] = macdsignal_sp
    df_sp500['macdhist_S'] = macdhist_sp


    # add MFI - money flow index to past 10 days
    df['Volume'] = df['Volume'].astype('float64')
    df_nasdaq['Volume'] = df_nasdaq['Volume'].astype('float64')
    df_sp500['Volume'] = df_sp500['Volume'].astype('float64')
    MFI_numpy_df = tb.MFI(df['High'].to_numpy(), df['Low'].to_numpy(),
                              df['Close'].to_numpy(),df['Volume'].to_numpy(), timeperiod=14)
    df['MFI'] = MFI_numpy_df
    MFI_numpy_dfnasdaq = tb.MFI(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                df_nasdaq['Close'].to_numpy(), df_nasdaq['Volume'].to_numpy(),timeperiod=14)
    df_nasdaq['MFI_N'] = MFI_numpy_dfnasdaq
    MFI_numpy_dfSP500 = tb.MFI(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                               df_sp500['Close'].to_numpy(), df_sp500['Volume'].to_numpy(),timeperiod=14)
    df_sp500['MFI_S'] = MFI_numpy_dfSP500

    # add ADX14 - average directional Movement index to past 14 days
    ADX14_numpy_df = tb.ADX(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=14)
    df['ADX14'] = ADX14_numpy_df
    ADX14_numpy_dfnasdaq = tb.ADX(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                  df_nasdaq['Close'].to_numpy(), timeperiod=14)
    df_nasdaq['ADX14_N'] = ADX14_numpy_dfnasdaq
    ADX14_numpy_dfSP500 = tb.ADX(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                                 df_sp500['Close'].to_numpy(), timeperiod=14)
    df_sp500['ADX14_S'] = ADX14_numpy_dfSP500

    # add MOM1 - momentum
    MOM1_numpy_df = tb.MOM(df['Close'].to_numpy(), timeperiod=1)
    df['MOM1'] = MOM1_numpy_df
    MOM1_numpy_dfnasdaq = tb.MOM(df_nasdaq['Close'].to_numpy(), timeperiod=1)
    df_nasdaq['MOM1_N'] = MOM1_numpy_dfnasdaq
    MOM1_numpy_dfSP500 = tb.MOM(df_sp500['Close'].to_numpy(), timeperiod=1)
    df_sp500['MOM1_S'] = MOM1_numpy_dfSP500

    # add ROCR12 - rate of change ratio
    ROCR12_numpy_df = tb.ROCR(df['Close'].to_numpy(), timeperiod=12)
    df['ROCR12'] = ROCR12_numpy_df
    ROCR12_numpy_dfnasdaq = tb.ROCR(df_nasdaq['Close'].to_numpy(), timeperiod=12)
    df_nasdaq['ROCR12_N'] = ROCR12_numpy_dfnasdaq
    ROCR12_numpy_dfSP500 = tb.ROCR(df_sp500['Close'].to_numpy(), timeperiod=12)
    df_sp500['ROCR12_S'] = ROCR12_numpy_dfSP500

    # add ATR - average true range
    ATR_numpy_df = tb.ATR(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=14)
    df['ATR'] = ATR_numpy_df
    ATR_numpy_dfnasdaq = tb.ATR(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                df_nasdaq['Close'].to_numpy(), timeperiod=14)
    df_nasdaq['ATR_N'] = ATR_numpy_dfnasdaq
    ATR_numpy_dfSP500 = tb.ATR(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                               df_sp500['Close'].to_numpy(), timeperiod=14)
    df_sp500['ATR_S'] = ATR_numpy_dfSP500

    # ADD OBV - on balance volume
    OBV_numpy_df = tb.OBV(df['Close'].to_numpy(), df['Volume'].to_numpy())
    df['OBV'] = OBV_numpy_df
    OBV_numpy_dfnasdaq = tb.OBV(df_nasdaq['Close'].to_numpy(), df_nasdaq['Volume'].to_numpy())
    df_nasdaq['OBV_N'] = OBV_numpy_dfnasdaq
    OBV_numpy_dfSP500 = tb.OBV(df_sp500['Close'].to_numpy(), df_sp500['Volume'].to_numpy())
    df_sp500['OBV_S'] = OBV_numpy_dfSP500

    # add TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    TRIX_numpy_df = tb.TRIX(df['Close'].to_numpy(), timeperiod=30)
    df['TRIX'] = TRIX_numpy_df
    TRIX_numpy_dfnasdaq = tb.TRIX(df_nasdaq['Close'].to_numpy(),timeperiod=30)
    df_nasdaq['TRIX_N'] = TRIX_numpy_dfnasdaq
    TRIX_numpy_dfSP500 = tb.TRIX(df_sp500['Close'].to_numpy(),timeperiod=30)
    df_sp500['TRIX_S'] = TRIX_numpy_dfSP500

    # add CCI20 - commodity channel index for the past 20 days
    CCI20_numpy_df = tb.CCI(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=20)
    df['CCI20'] = CCI20_numpy_df
    CCI20_numpy_dfnasdaq = tb.CCI(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                df_nasdaq['Close'].to_numpy(), timeperiod=20)
    df_nasdaq['CCI20_N'] = CCI20_numpy_dfnasdaq
    CCI20_numpy_dfSP500 = tb.CCI(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                                 df_sp500['Close'].to_numpy(), timeperiod=20)
    df_sp500['CCI20_S'] = CCI20_numpy_dfSP500

    # add ADX20 - average directional Movement index to past 14 days
    ADX20_numpy_df = tb.ADX(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), timeperiod=20)
    df['ADX20'] = ADX20_numpy_df
    ADX20_numpy_dfnasdaq = tb.ADX(df_nasdaq['High'].to_numpy(), df_nasdaq['Low'].to_numpy(),
                                  df_nasdaq['Close'].to_numpy(), timeperiod=20)
    df_nasdaq['ADX20_N'] = ADX20_numpy_dfnasdaq
    ADX20_numpy_dfSP500 = tb.ADX(df_sp500['High'].to_numpy(), df_sp500['Low'].to_numpy(),
                                 df_sp500['Close'].to_numpy(), timeperiod=20)
    df_sp500['ADX20_S'] = ADX20_numpy_dfSP500

    # add MOM3 - momentum
    MOM3_numpy_df = tb.MOM(df['Close'].to_numpy(), timeperiod=3)
    df['MOM3'] = MOM3_numpy_df
    MOM3_numpy_dfnasdaq = tb.MOM(df_nasdaq['Close'].to_numpy(), timeperiod=3)
    df_nasdaq['MOM3_N'] = MOM3_numpy_dfnasdaq
    MOM3_numpy_dfSP500 = tb.MOM(df_sp500['Close'].to_numpy(), timeperiod=3)
    df_sp500['MOM3_S'] = MOM3_numpy_dfSP500

    # add ROCR20 - rate of change ratio
    ROCR3_numpy_df = tb.ROCR(df['Close'].to_numpy(), timeperiod=3)
    df['ROCR3'] = ROCR3_numpy_df
    ROCR3_numpy_dfnasdaq = tb.ROCR(df_nasdaq['Close'].to_numpy(), timeperiod=3)
    df_nasdaq['ROCR3_N'] = ROCR3_numpy_dfnasdaq
    ROCR3_numpy_dfSP500 = tb.ROCR(df_sp500['Close'].to_numpy(), timeperiod=3)
    df_sp500['ROCR3_S'] = ROCR3_numpy_dfSP500

    # add RSI6 - relative strength index for the past 14 days
    RSI6_numpy_df = tb.RSI(df['Close'].to_numpy(), timeperiod=6)
    df['RSI6'] = RSI6_numpy_df
    RSI6_numpy_dfnasdaq = tb.RSI(df_nasdaq['Close'].to_numpy(), timeperiod=6)
    df_nasdaq['RSI6_N'] = RSI6_numpy_dfnasdaq
    RSI6_numpy_dfSP500 = tb.RSI(df_sp500['Close'].to_numpy(), timeperiod=6)
    df_sp500['RSI6_S'] = RSI6_numpy_dfSP500

    # add BBANDS - Bollinger Bands
    upperband, middleband, lowerband = tb.BBANDS(df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['upperband'] = upperband
    df['middleband'] = middleband
    df['lowerband'] = lowerband
    upperbandna, middlebandna, lowerbandna = tb.BBANDS(df_nasdaq['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df_nasdaq['upperbandna_N'] = upperbandna
    df_nasdaq['middlebandna_N'] = middlebandna
    df_nasdaq['lowerbandna_N'] = lowerbandna
    upperbandsp, middlebandsp, lowerbandsp = tb.BBANDS(df_sp500['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df_sp500['upperbandsp_S'] = upperbandsp
    df_sp500['middlebandsp_S'] = middlebandsp
    df_sp500['lowerbandsp_S'] = lowerbandsp

    # add EMA6 - Exponential Moving Average
    EMA6_numpy_df = tb.EMA(df['Close'].to_numpy(), timeperiod=6)
    df['EMA6'] = EMA6_numpy_df
    EMA6_numpy_dfnasdaq = tb.EMA(df_nasdaq['Close'].to_numpy(), timeperiod=6)
    df_nasdaq['EMA6_N'] = EMA6_numpy_dfnasdaq
    EMA6_numpy_dfSP500 = tb.EMA(df_sp500['Close'].to_numpy(), timeperiod=6)
    df_sp500['EMA6_S'] = EMA6_numpy_dfSP500

    # add EMA12 - Exponential Moving Average
    EMA12_numpy_df = tb.EMA(df['Close'].to_numpy(), timeperiod=6)
    df['EMA12'] = EMA12_numpy_df
    EMA12_numpy_dfnasdaq = tb.EMA(df_nasdaq['Close'].to_numpy(), timeperiod=6)
    df_nasdaq['EMA12_N'] = EMA12_numpy_dfnasdaq
    EMA12_numpy_dfSP500 = tb.EMA(df_sp500['Close'].to_numpy(), timeperiod=6)
    df_sp500['EMA12_S'] = EMA12_numpy_dfSP500

    # add TSF10 - Time Series Forecast FROM PAST 10DAYS
    TSF10_numpy_df = tb.TSF(df['Close'].to_numpy(), timeperiod=10)
    df['TSF10'] = TSF10_numpy_df
    TSF10_numpy_dfnasdaq = tb.TSF(df_nasdaq['Close'].to_numpy(), timeperiod=10)
    df_nasdaq['TSF10_N'] = TSF10_numpy_dfnasdaq
    TSF10_numpy_dfSP500 = tb.TSF(df_sp500['Close'].to_numpy(), timeperiod=10)
    df_sp500['TSF10_S'] = TSF10_numpy_dfSP500

    # add TSF20 - Time Series Forecast FROM PAST 20DAYS
    TSF20_numpy_df = tb.TSF(df['Close'].to_numpy(), timeperiod=20)
    df['TSF20'] = TSF20_numpy_df
    TSF20_numpy_dfnasdaq = tb.TSF(df_nasdaq['Close'].to_numpy(), timeperiod=20)
    df_nasdaq['TSF20_N'] = TSF20_numpy_dfnasdaq
    TSF20_numpy_dfSP500 = tb.TSF(df_sp500['Close'].to_numpy(), timeperiod=20)
    df_sp500['TSF20_S'] = TSF20_numpy_dfSP500

    # add SMA3 - simple moving average for past 3 days
    SMA3_numpy_df = tb.SMA(df['Close'].to_numpy(), timeperiod=3)
    df['SMA3'] = SMA3_numpy_df
    SMA3_numpy_dfnasdaq = tb.SMA(df_nasdaq['Close'].to_numpy(), timeperiod=3)
    df_nasdaq['SMA3_N'] = SMA3_numpy_dfnasdaq
    SMA3_numpy_dfSP500 = tb.SMA(df_sp500['Close'].to_numpy(), timeperiod=3)
    df_sp500['SMA3_P'] = SMA3_numpy_dfSP500

    # add SMA - helper column for function labelling
    SMA_numpy_df = tb.SMA(df['Close'].to_numpy(), timeperiod=ndays)
    df['SMA'] = SMA_numpy_df
    
    df.to_csv('df_popedfeatures.csv')
    return df, df_nasdaq,df_sp500


def integratedframes(df_poped, df_nasdaq_poped,df_sp500_poped):
    """
    Integrate three input dataframes and impute missing values, return a integrated single dataframe
    :param df_poped: df with all indicators populated
    :param df_nasdaq_poped: df_nasdaq with all indicators populated
    :param df_sp500_poped: df_sp500 with all indicators populated
    :return:  an integrated single dataframe
    """
    df_nasdaq_poped.rename(columns = {'Open':'Open_N', 'High':'High_N', 'Low':'Low_N', 'Close':'Close_N','Volume': 'Volume_N'},inplace=True)
    df_sp500_poped.rename(columns = {'Open':'Open_S', 'High':'High_S', 'Low':'Low_S', 'Close':'Close_S','Volume': 'Volume_S'},inplace=True)
    df_dropcolumns = ['Open', 'High', 'Low']
    df_nasdaq_dropcolumns = ['Open_N', 'High_N', 'Low_N']
    df_sp500_dropcolumns = ['Open_S', 'High_S', 'Low_S']
    df_dropped = df_poped.drop(df_dropcolumns, axis=1)
    dfnasdaq_deopped = df_nasdaq_poped.drop(df_nasdaq_dropcolumns, axis=1)
    dfsp500_dropped = df_sp500_poped.drop(df_sp500_dropcolumns, axis=1)
    try:
        concatenated = pd.concat([pd.concat([df_dropped, dfnasdaq_deopped], axis=1), dfsp500_dropped], axis=1)

    except AttributeError:
        print("check whether the shape of three dataframes match!")
    else:
        concatenated_nadropped = concatenated.dropna(axis = 0, how='any')
        return concatenated_nadropped


def labelling(concatenated_df, ndays=NDAYS_LOOKAHEAD):
    """
    Label the concatenated_df according to the following rules:
    Y(t) = 1 if SMA(t+3)>SMA(t); Y(t) = -1 if SMA(t+3) < Price(t)  (assume we want to predict the (ndays) days look-ahead
    trend for the stock)
    :param concatenated_df: the df calculated from function integratedframes
    :param ndays: an integer representing how many days look-ahead we want to predict
    :return: the final labeled dataframe ready for feature selection
    """
    concatenated_df['label'] = 0
    for i in range(concatenated_df.shape[0]):
        timestampList = list(concatenated_df.index)
        if (i+ndays < concatenated_df.shape[0]):
            starttime = str(timestampList[i])[0:10]
            endtime = str(timestampList[i + ndays])[0:10]
            if (concatenated_df.loc[endtime, 'SMA'] > concatenated_df.loc[starttime, 'SMA']):
                concatenated_df.loc[starttime, 'label'] = 1

    final_df = concatenated_df.dropna(axis = 0, how='any')
    final_df = final_df.drop(['SMA'],axis = 1)
    #TODO: 真实预测的时候这里最近几天的数据要加上，别删了，测试模型的时候必须要label所以暂时删了
    final_df = final_df.iloc[:(concatenated_df.shape[0] - ndays), :]
    return final_df



def labelling_realtime(concatenated_df,ndays=NDAYS_LOOKAHEAD):
    """
    Label the concatenated_df according to the following rules:
    Y(t) = 1 if SMA(t+3)>SMA(t); Y(t) = -1 if SMA(t+3) < Price(t) (assume we want to predict the (ndays, here is 3) days look-ahead
    trend for the stock)
    :param concatenated_df: the df calculated from function integratedframes
    :param ndays: an integer representing how many days look-ahead we want to predict
    :return: the final labeled dataframe ready for feature selection (note that the only difference between this function and
    function labelling is that this one does not delete the last (ndays) rows from the resulting dataframe due to
    our prediction method. If we delete those, this function will be exactly the same as function labelling.)
    """
    concatenated_df['label'] = 0
    for i in range(concatenated_df.shape[0]):
        timestampList = list(concatenated_df.index)
        if (i + ndays < concatenated_df.shape[0]):
            starttime = str(timestampList[i])[0:10]
            endtime = str(timestampList[i + ndays])[0:10]
            if (concatenated_df.loc[endtime, 'SMA'] > concatenated_df.loc[starttime, 'SMA']):
                concatenated_df.loc[starttime, 'label'] = 1

    final_df = concatenated_df.dropna(axis=0, how='any')
    final_df = final_df.drop(['SMA'], axis=1)
    return final_df


def prepare_data(stocknameabrev, ndays):
    """
        API KEY(Quandl): Jaq5cjYcrvukouwWEqwA
        perform everything in module prepare_data
        :param: stocknameabrev: a string represents a stock in American stock exchanges e.g 'AAPL.US'
        :param: enddatetime: a datetime instance represents the last trading day we want to fetch data
        :return: a dataframe ready for feature selection.
    """
    df, df_nasdaq, df_sp500 = readdata(stocknameabrev)
    df_poped, df_nasdaq_poped, df_sp500_poped = popfeatures(df, df_nasdaq, df_sp500, ndays=ndays)
    concatenated = integratedframes(df_poped, df_nasdaq_poped, df_sp500_poped)
    finaldf = labelling(concatenated, ndays=ndays)
    return finaldf



def prepare_data_realtime(stocknameabrev, ndays=NDAYS_LOOKAHEAD):
    """
        API KEY(Quandl): Jaq5cjYcrvukouwWEqwA
        perform everything in module prepare_data
        :param: stocknameabrev: a string represents a stock in American stock exchanges e.g 'AAPL.US'
        :param: enddatetime: a datetime instance represents the last trading day we want to fetch data
        :return: a dataframe for real time prediction.
    """
    df, df_nasdaq, df_sp500 = readdata(stocknameabrev)
    df_poped, df_nasdaq_poped, df_sp500_poped = popfeatures(df, df_nasdaq, df_sp500, ndays=ndays)
    concatenated = integratedframes(df_poped, df_nasdaq_poped, df_sp500_poped)
    finaldf = labelling_realtime(concatenated, ndays=ndays)
    return finaldf