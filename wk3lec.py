

# #EXERCISE 2
# import pandas  as pd
#
# # Get the housing data
# df = pd.read_csv('data/housing_classification.csv')
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(df.head(5))
#
# # Split into two sets
# y = df['price']
# X = df.drop('price', axis=1)
#
# from sklearn.ensemble        import BaggingClassifier
# from sklearn.neighbors       import KNeighborsClassifier
# from sklearn.linear_model    import RidgeClassifier
# from sklearn.svm             import SVC
# from sklearn.metrics import classification_report
# from   sklearn.linear_model    import LogisticRegression
#
#
# # Create classifiers
# knn         = KNeighborsClassifier()
# svc         = SVC()
# rg          = RidgeClassifier()
# lr = LogisticRegression(fit_intercept=True, solver='liblinear')
#
# # Build array of classifiers.
# classifierArray   = [knn, svc, rg, lr]
#
# def showStats(classifier, scores):
#     print(classifier + ":    ", end="")
#     strMean = str(round(scores.mean(),2))
#
#     strStd  = str(round(scores.std(),2))
#     print("Mean: "  + strMean + "   ", end="")
#     print("Std: " + strStd)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#
# def evaluateModel(model, X_test, y_test, title):
#     print("\n*** " + title + " ***")
#     predictions = model.predict(X_test)
#     report = classification_report(y_test, predictions)
#     print(report)
#
# # Search for the best classifier.
# for clf in classifierArray:
#     modelType = clf.__class__.__name__
#
#     # Create and evaluate stand-alone model.
#     clfModel    = clf.fit(X_train, y_train)
#     evaluateModel(clfModel, X_test, y_test, modelType)
#
#     # max_features means the maximum number of features to draw from X.
#     # max_samples sets the percentage of available data used for fitting.
#     bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=6,
#                                     n_estimators=100)
#     baggedModel = bagging_clf.fit(X_train, y_train)
#     evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)
#


#
# #EXERCISE 3
# import pandas as pd
# from sklearn.ensemble     import BaggingRegressor
# from sklearn.linear_model import LinearRegression
#
# from   sklearn.model_selection import train_test_split
# import numpy as np
# from   sklearn.metrics         import mean_squared_error
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# # Load and prepare data.
# FOLDER  = 'data/'
# FILE    = 'petrol_consumption.csv'
# dataset = pd.read_csv(FOLDER + FILE)
# print(dataset)
# X = dataset.copy()
# del X['Petrol_Consumption']
# y = dataset[['Petrol_Consumption']]
#
# feature_combo_list = []
# def evaluateModel(model, X_test, y_test, title, num_estimators, max_features):
#     print("\n****** " + title)
#     predictions = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#
#     # Store statistics and add to list.
#     stats = {"type":title, "rmse":rmse,
#              "estimators":num_estimators, "features":max_features}
#     feature_combo_list.append(stats)
#
# num_estimator_list = [750, 800, 900, 1000]
# max_features_list  = [0.2, 0.4, 0.6]
#
# for num_estimators in num_estimator_list:
#     for max_features in max_features_list:
#         # Create random split.
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#         import numpy as np
#
#         # Build linear regression ensemble.
#         ensembleModel = BaggingRegressor(estimator=LinearRegression(),
#                                 max_features=max_features,
#                                 # Can be percent (float) or actual
#                                 # total of samples (int).
#                                 max_samples =1,
#                                 n_estimators=num_estimators).fit(X_train, y_train.values.ravel())
#         evaluateModel(ensembleModel, X_test, y_test, "Ensemble",
#                       num_estimators, max_features)
#
#         # Build stand alone linear regression model.
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         evaluateModel(model, X_test, y_test, "Linear Regression", None, None)
#
# # Build data frame with dictionary objects.
# dfStats = pd.DataFrame()
# print(dfStats)
# for combo in feature_combo_list:
#     dfStats = pd.concat([dfStats,
#                          pd.DataFrame.from_records([combo])],
#                          ignore_index=True)
#
# # Sort and show all combinations.
# # Show all rows
# pd.set_option('display.max_rows', None)
# dfStats = dfStats.sort_values(by=['type', 'rmse'])
# print(dfStats)


# #Exercise5
# import pandas  as pd
# from sklearn.metrics import classification_report
#
# # Get the housing data
# df = pd.read_csv('data/housing_classification.csv')
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(df.head(5))
#
# # Split into two sets
# y = df['price']
# X = df.drop('price', axis=1)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#
# from sklearn.model_selection import cross_val_score
# from mlxtend.classifier      import EnsembleVoteClassifier
# from xgboost                 import XGBClassifier, plot_importance
# from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
# from   sklearn.linear_model    import LogisticRegression
# lr = LogisticRegression(fit_intercept=True, solver='liblinear')
#
# ada_boost   = AdaBoostClassifier()
# grad_boost  = GradientBoostingClassifier()
# xgb_boost   = XGBClassifier()
# classifiers = [ada_boost, grad_boost, xgb_boost, lr]
#
# for clf in classifiers:
#     print(clf.__class__.__name__)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     report = classification_report(y_test, predictions)
#     print(report)
#



# #EXERCISE 6
# import pandas as pd
# # Get the housing data
# df = pd.read_csv('data/iris_v2.csv')
#
# dict_map = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
# df['target'] = df['iris_type'].map(dict_map)
#
# y = df['target']
# X = df.copy()
# del X['target']
# del X['iris_type']
#
#
# import pandas  as pd
# from sklearn.metrics import classification_report
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(df.head(5))
#
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#
# from sklearn.model_selection import cross_val_score
# from mlxtend.classifier      import EnsembleVoteClassifier
# from xgboost                 import XGBClassifier, plot_importance
# from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
#
# ada_boost   = AdaBoostClassifier()
# grad_boost  = GradientBoostingClassifier()
# xgb_boost   = XGBClassifier()
# classifiers = [ada_boost, grad_boost, xgb_boost]
#
# for clf in classifiers:
#     print(clf.__class__.__name__)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     report = classification_report(y_test, predictions)
#     print(report)



#Exercise7
import pandas  as pd
from sklearn.metrics import classification_report

# Get the housing data
df = pd.read_csv('data/housing_classification.csv')
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))

# Split into two sets
y = df['price']
X = df.drop('price', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.model_selection import cross_val_score
from mlxtend.classifier      import EnsembleVoteClassifier
from xgboost                 import XGBClassifier, plot_importance
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier

ada_boost   = AdaBoostClassifier()
grad_boost  = GradientBoostingClassifier()
xgb_boost   = XGBClassifier()
eclf        = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost,
                                           xgb_boost], voting='hard')
classifiers = [ada_boost, grad_boost, xgb_boost, eclf]


for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
