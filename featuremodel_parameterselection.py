from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from prepare_data import prepare_data
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import datetime
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

finaldf = prepare_data('AAPL.US', 5)
X = finaldf.iloc[:,:-1]
y = finaldf.iloc[:,-1]
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)


paramgrid = {"n_estimators": [250,300,350,400,450,500],
             "criterion"     : ['gini','entropy'],
             "max_depth" : [None, 6,7,8,9,10],
             'min_samples_split' : [80,85,90,95,100],
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
    print(type(best.feature_importances_))
    return best.feature_importances_
