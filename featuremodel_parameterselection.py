from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from prepare_data import *
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import datetime
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

finaldf = prepare_data(STOCK_CODE, NDAYS_LOOKAHEAD)
X = finaldf.iloc[:,:-1]
y = finaldf.iloc[:,-1]
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)


paramgrid = {"n_estimators": [150,250,350,400,450,500,600],
             "criterion"     : ['gini','entropy'],
             "max_depth" : [1,2,3,4,5,6,7,8,9,10],
             'min_samples_split' : [80,90,100,150,180,190,200],
             }


def show_featureimportance():
    cv = GridSearchCV(estimator=ExtraTreesClassifier(),
                      param_grid=paramgrid,
                      scoring="accuracy",
                      cv=TimeSeriesSplit(n_splits=4),
                      verbose=1,
                      n_jobs=5,
                      return_train_score=True,
                      refit=True)
    cv.fit(X, y)
    best = cv.best_estimator_
    bestscore = cv.best_score_
    print(best, bestscore)
    return best.feature_importances_
