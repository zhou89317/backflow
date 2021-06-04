from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from feature_selection import preparedata_forSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

X_train, X_test, y_train, y_test = preparedata_forSVM()
X_train.to_csv('X_TRAIN_SVMTEST.csv')
X_test.to_csv('X_TEST_SVMTEST.csv')
y_train.to_csv('y_TRAIN_SVMTEST.csv')
y_test.to_csv('y_TEST_SVMTEST.csv')

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
                      return_train_score=True,
                      refit=True)

cv.fit(X_train, y_train)
print("cv results: ")
print(pd.DataFrame(cv.cv_results_))
best = cv.best_estimator_
bestscore = cv.best_score_
y_pred = best.predict(X_test)
print(best, bestscore)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_pred=y_pred, y_true=y_test))