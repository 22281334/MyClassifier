import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 10-fold cross valiadation
cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


# K-Nearest Neighbour
def kNNClassifier(X, y, K):
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:0.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Logistic Regression
def logregClassifier(X, y):
    logreg = LogisticRegression()
    scores = cross_val_score(logreg, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:0.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Naive Bayes
def nbClassifier(X, y):
    nb = GaussianNB()
    scores = cross_val_score(nb, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:0.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Decision Tree
def dtClassifier(X, y):
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    scores = cross_val_score(dt, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:0.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Bagging
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0),
                                n_estimators=n_estimators, max_samples=max_samples, random_state=0)
    scores = cross_val_score(bag_clf, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Ada Boost
def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0),
                                 n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(ada_clf, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Gradient Boosting
def gbClassifier(X, y, n_estimators, learning_rate):
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(gb_clf, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')
    return scores, scores.mean()


# Linear SVM
def bestLinClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(kernel="linear", random_state=0), param_grid, cv=cvKFold, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['C'])
    print(grid_search.best_params_['gamma'])
    print("{:.4f}".format(grid_search.best_score_))
    print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')


# Random Forest
def bestRFClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'n_estimators': [10, 20, 50, 100],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_leaf_nodes': [10, 20, 30]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=0, criterion='entropy'), param_grid, cv=cvKFold,
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['n_estimators'])
    print(grid_search.best_params_['max_features'])
    print(grid_search.best_params_['max_leaf_nodes'])
    print("{:.4f}".format(grid_search.best_score_))
    print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')


# Label Encoder
def labelEncode(classes):
    labels = np.unique(classes)
    labelEncode = LabelEncoder()
    labelEncode.fit(labels)
    return labelEncode.transform(classes)


# Read the config file
def conf_file(file):
    conf = pd.read_csv(file)
    # Convert parameters to a list:
    parameters = conf.iloc[0].tolist()
    return parameters


# print data
def printData():
    for i in range(len(dataList)):
        for j in dataList[i]:
            print(j, end=',')
        if i < len(dataList) - 1:
            print(labelEncoder[i])
        else:
            print(labelEncoder[i], end='')


# Read csv file
dataset = pd.read_csv(sys.argv[1])

# Replace missing attribute values "?" to np.nan
dataset = dataset.replace("?", np.nan)
data = dataset.iloc[:, 0:-1]

# fit and transform np.nan to mean value
impMean = SimpleImputer(missing_values=np.nan, strategy='mean')
data = impMean.fit_transform(data)

# Normalisation of each attribute value to to normalise the values between [0,1]
data = MinMaxScaler().fit_transform(data)

classes = dataset.iloc[:, -1].tolist()
labelEncoder = labelEncode(classes).astype(np.int)

# Add all values to data list and keep 4 decimals
dataList = []
for i in data:
    temp = []
    for j in i:
        temp.append('%0.4f' % j)
    dataList.append(temp)

if sys.argv[2] == 'NN':
    parameter_list = conf_file(sys.argv[3])
    K = int(parameter_list[0])
    kNNClassifier(dataList, labelEncoder, K)
elif sys.argv[2] == 'LR':
    logregClassifier(dataList, labelEncoder)
elif sys.argv[2] == 'NB':
    nbClassifier(dataList, labelEncoder)
elif sys.argv[2] == 'DT':
    dtClassifier(dataList, labelEncoder)
elif sys.argv[2] == 'BAG':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    max_samples = int(parameter_list[1])
    max_depth = int(parameter_list[2])
    bagDTClassifier(dataList, labelEncoder, n_estimators, max_samples, max_depth)
elif sys.argv[2] == 'ADA':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    learning_rate = parameter_list[1]
    max_depth = int(parameter_list[2])
    adaDTClassifier(dataList, labelEncoder, n_estimators, learning_rate, max_depth)
elif sys.argv[2] == 'GB':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    learning_rate = parameter_list[1]
    gbClassifier(dataList, labelEncoder, n_estimators, learning_rate)
elif sys.argv[2] == 'RF':
    bestRFClassifier(dataList, labelEncoder)
elif sys.argv[2] == 'SVM':
    bestLinClassifier(dataList, labelEncoder)
elif sys.argv[2] == 'P':
    printData()
