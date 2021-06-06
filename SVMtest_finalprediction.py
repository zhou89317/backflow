from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from feature_selection import preparedata_forSVM, prepare_data_realtime,apply_selectfeatures_realtime
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from prepare_data import NDAYS_LOOKAHEAD, STOCK_CODE

X_train, X_test, y_train, y_test,columns_selected = preparedata_forSVM()
# X_train.to_csv('X_TRAIN_SVMTEST.csv')
# X_test.to_csv('X_TEST_SVMTEST.csv')
# y_train.to_csv('y_TRAIN_SVMTEST.csv')
# y_test.to_csv('y_TEST_SVMTEST.csv')

scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

paramgrid = {'C' : [1,2,3,4,5],
             "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
             "gamma"     : ['scale','auto']
             }

SVCclassfier = SVC()

cv = GridSearchCV(estimator=SVCclassfier,
                      param_grid=paramgrid,
                      scoring="accuracy",
                      cv=TimeSeriesSplit(n_splits=10),
                      verbose=1,
                      n_jobs=5,
                      refit=True)

cv.fit(X_train, y_train)
best = cv.best_estimator_
bestscore = cv.best_score_
y_pred = best.predict(X_test)
bestparamsdict = cv.best_params_
best_estimator_string = "The best estimator for the data set is {}".format(best)
best_estimatorpropeprty_string = "The crossvalidation score, test accuracy, hyperparameter settings for the best estimators" \
                                 "is: {}, {}, {} respectively.".format(bestscore,accuracy_score(y_test,y_pred), bestparamsdict)
best_confusion = "The test confusion matrix for the best hyperparameter settings is: {} ".format(confusion_matrix(y_pred=y_pred, y_true=y_test))
print(best_estimator_string)
print(best_estimatorpropeprty_string)


finaldf = prepare_data_realtime(STOCK_CODE, NDAYS_LOOKAHEAD)
Xreal = finaldf.iloc[:, :-1]
yreal = finaldf.iloc[:,-1]
Xreal = Xreal.iloc[:, columns_selected]
X_trainreal = Xreal.iloc[:(Xreal.shape[0] - NDAYS_LOOKAHEAD), :]
y_trainreal = yreal.iloc[:(yreal.shape[0] - NDAYS_LOOKAHEAD)]
X_topredreal = Xreal.iloc[-NDAYS_LOOKAHEAD:]
# print("X_topredreal:")
print(X_topredreal)



finalSVMclassifier = SVC(C=bestparamsdict['C'], gamma=bestparamsdict['gamma'], kernel=bestparamsdict['kernel'])
finalSVMclassifier.fit(X_trainreal,y_trainreal)
y_realpred = finalSVMclassifier.predict(X_topredreal)
print("The {} - days ahead return trend of {} for the latest {} days are: {}".format(NDAYS_LOOKAHEAD,
                                                                                     STOCK_CODE,
                                                                                     NDAYS_LOOKAHEAD,
                                                                                     y_realpred))
date_string_begin_predict = str(list(X_topredreal.index)[-1])
result_statement = "Standing at {date}, the trend of {stock} for next {num} days is {updown}".format(
                    date = date_string_begin_predict,
                    num = NDAYS_LOOKAHEAD,
                    updown = 'up' if y_realpred[-1]==1 else 'down',
                    stock = STOCK_CODE)
print(result_statement)