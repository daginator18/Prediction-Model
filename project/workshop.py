# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue May 28 17:16:07 2024

# @author: hampusstalhandske
# """

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# #%%
# iris = load_iris()
# #%%
# plt.scatter(iris.data[:,0], iris.data[:,1],
#              c = iris.target,
#              s = 50,
#              cmap = "rainbow")

# plt.xlabel("sepal_length")
# plt.ylabel("sepal_width")
# #%%
# plt.scatter(iris.data[:,2], iris.data[:,3],
#              c = iris.target,
#              s = 50,
#              cmap = "rainbow")

# plt.xlabel("petal_length")
# plt.ylabel("petal_width")
# #%%

# X = pd.DataFrame(iris.data[:,:2], columns=iris.feature_names[:2])
# y = iris.target

# clf = DecisionTreeClassifier(random_state=42, max_depth=2)

# clf.fit(X, y)

# y_pred = clf.predict(X)

# accuracy_dt = clf.score(X, y)
# print(accuracy_dt)

# plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], impurity=True)
# #%%
# conf_mat = confusion_matrix(y, y_pred)
# sns.heatmap(conf_mat, xticklabels=iris.target_names, yticklabels=iris.target_names,
#             annot=True)
# #%%
# X = pd.DataFrame(iris.data[:,2:], columns=iris.feature_names[2:])
# y = iris.target

# clf = DecisionTreeClassifier(random_state=42, max_depth=2)

# clf.fit(X, y)

# y_pred = clf.predict(X)

# accuracy_dt = clf.score(X, y)
# print(accuracy_dt)

# plot_tree(clf, filled=True, feature_names=iris.feature_names[2:], impurity=True)
# #%%

# conf_mat = confusion_matrix(y, y_pred)
# sns.heatmap(conf_mat, xticklabels=iris.target_names, yticklabels=iris.target_names,
#             annot=True)

# #%%
# file = r'titanic data'
# titanic = pd.read_csv(file)
# #%%

# data = titanic.dropna()

# X = data.drop("survived", axis=1)
# X = pd.get_dummies(X)

# y = data["survived"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #%%

# dt_classifier = DecisionTreeClassifier(random_state=42)

# dt_classifier.fit(X_train, y_train)

# accuracy_dt = dt_classifier.score(X_test, y_test)

# print(accuracy_dt)

# #%%

# from sklearn.ensemble import RandomForestClassifier

# random_forest = RandomForestClassifier(random_state=42)

# random_forest.fit(X_train, y_train)

# accuracy_rf = random_forest.score(X_test, y_test)

# print(accuracy_rf)

# #%%

# param_grid = {
#     "max_features" : [2,3,4,5],
#     "min_samples_split" : [2, 3, 4, 5, 6],
#     "max_depth" : [2,3,4,5,6,7]
#     }

# search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5,
#                       verbose=1)

# search.fit(X_train, y_train)

# print("Best CV score: {} using {}".format(search.best_score_, search.best_params_))


# #%%
# param_grid = {
#     "n_estimators" : [100],
#     "max_features" : [3,4,5],
#     "min_samples_split" : [4, 5, 6],
#     "max_depth" : [4,6,7]
#     }

# search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5,
#                       verbose=1)

# search.fit(X_train, y_train)

# print("Best CV score: {} using {}".format(search.best_score_, search.best_params_))

# #%%

# dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5, max_features=5, min_samples_split=5)

# dt_classifier.fit(X_train, y_train)

# accuracy_dt = dt_classifier.score(X_test, y_test)

# print(accuracy_dt)

# #%%

# random_forest = RandomForestClassifier(random_state=42, max_depth=7, max_features=4, 
#                                        min_samples_split=4, n_estimators=100)

# random_forest.fit(X_train, y_train)

# accuracy_rf = random_forest.score(X_test, y_test)

# print(accuracy_rf)

# #%%

# feature_df = pd.DataFrame(titanic.columns.delete(0))
# feature_df.columns = ["Feature"]
# feature_df["Importance"] = pd.Series(random_forest.feature_importances_)

# feature_df

# #%%

# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

# param_grid = {
#     "n_neighbors" : range(1, 40)
#     }

# search = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, verbose=1)

# X_trainn = X_train*1/np.max(np.abs(X_train), axis=0)
# X_testn = X_test*1/np.max(np.abs(X_train), axis=0)


# search.fit(X_trainn, y_train)

# print("Best CV score: {} using {}".format(search.best_score_, search.best_params_))

# #%%

# knn = KNeighborsClassifier(n_neighbors=28)

# knn.fit(X_trainn, y_train)

# accuracy_knn = knn.score(X_testn, y_test)

# print(accuracy_knn)












