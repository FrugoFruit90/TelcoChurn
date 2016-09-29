import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc

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
# print(X.ix[2])
X = StandardScaler().fit_transform(X)
# print(X[0])

clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
clf.fit(X, y)
y_hat = clf.predict(X)
print('CV scores: %s' % np.mean(clf.scores_[1]))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_hat)
print('AUC: %f' % auc(false_positive_rate, true_positive_rate))
