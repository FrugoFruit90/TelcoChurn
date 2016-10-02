import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# TODO check if maybe roc_auc_score is better/ more relevant and possibly implement
# TODO model stacking (i.e. voting classifier)

# Load the data
X = pd.read_csv('churn.csv', ';')

# Extract the predicted variable
y = np.where(X['Churn?'] == 'True.', 1, 0)

# X is the feature space so drop the y column from it.
# Additionally, drop phone number as I have no reason to believe that it influences churn decision
# Finally, notice that there are only 3 area codes, all from California, but many more states.
# This means that this data is bad and should probably be dropped.
X.drop(['State', 'Area Code', 'Phone', 'Churn?'], axis=1, inplace=True)
X["Int'l Plan"] = X["Int'l Plan"] == 'yes'
X["VMail Plan"] = X["VMail Plan"] == 'yes'

# Some of the estimators require feature scaling to work properly
X = StandardScaler().fit_transform(X)

# Shuffle and divide the database into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Set
cv = 3

# Logistic Regression
lrc = LogisticRegressionCV(cv=cv, scoring='roc_auc', n_jobs=-1)
lrc.fit(X_train, y_train)
y_hat = lrc.predict(X_test)
print("Logistic regression")
print('CV scores: %s' % np.mean(lrc.scores_[1]))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# Support Vector Machines
# tuned_parameters = {'C': [1, 10, 100, 1000]}
# svmc = GridSearchCV(svm.SVC(kernel='linear', probability=True), param_grid=tuned_parameters, n_jobs=8, cv=cv)
# svmc.fit(X_train, y_train)
# y_hat = svmc.best_estimator_.predict(X_test)
# print("SVMs (linear kernel)")
# print('CV scores: %s' % np.mean(cross_val_score(svmc.best_estimator_, X_train, y_train, cv=cv)))
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
# print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# Random Forest
tuned_parameters = {"max_depth": [3, 7, 11, 15]}
rfc = GridSearchCV(RandomForestClassifier(n_estimators=200, n_jobs=-1), param_grid=tuned_parameters)
rfc.fit(X_train, y_train)
y_hat = rfc.best_estimator_.predict(X_test)
print("Random Forest")
print('CV scores: %s' % np.mean(cross_val_score(rfc.best_estimator_, X_train, y_train, cv=cv)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# Extra Trees Classifier
etc = GridSearchCV(ExtraTreesClassifier(n_estimators=200, n_jobs=-1), param_grid=tuned_parameters)
etc.fit(X_train, y_train)
y_hat = etc.best_estimator_.predict(X_test)
print("Extra Trees")
print('CV scores: %s' % np.mean(cross_val_score(etc, X_train, y_train, cv=cv)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# XGBoost
gbm = GridSearchCV(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05), param_grid=tuned_parameters)
gbm.fit(X_train, y_train)
y_hat = gbm.best_estimator_.predict(X_test)
print("XGBoost")
print('CV scores: %s' % np.mean(cross_val_score(gbm.best_estimator_, X_train, y_train, cv=cv)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# Stacked classifier
X_train2 = X_train[:]
X_test2 = X_test[:]
estimators = [rfc, etc, gbm]
for est in estimators:
    probability1 = est.best_estimator_.predict_proba(X_train)
    probability2 = est.best_estimator_.predict_proba(X_test)
    np.append(X_train2, probability1, axis=1)
    np.append(X_test2, probability2, axis=1)
lrc = LogisticRegressionCV(cv=cv, scoring='roc_auc', n_jobs=-1)
lrc.fit(X_train2, y_train)
y_hat = lrc.predict(X_test2)
print("Stacked classifier")
print('CV scores: %s' % np.mean(cross_val_score(lrc, X_train2, y_train, cv=cv)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))

# Voting classifier
X_train3 = X_train2[:, -3:]
X_test3 = X_test2[:, -3:]
vc = VotingClassifier(
    estimators=[('lr', rfc.best_estimator_), ('rf', etc.best_estimator_), ('gbm', gbm.best_estimator_)], voting='soft')
vc.fit(X_train3, y_train)
y_hat = vc.predict(X_test3)
print("Voting classifier")
print('CV scores: %s' % np.mean(cross_val_score(lrc, X_train3, y_train, cv=cv)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
print('AUC: %f \n' % auc(false_positive_rate, true_positive_rate))
