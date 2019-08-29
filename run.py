# #####################Import the packages
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sn
# from sklearn import preprocessing, svm
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
# from sklearn import metrics
#
# columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'Education num', 'Marital Status',
#            'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
#            'Hours/Week', 'Native country', 'Income']
# train = pd.read_csv('Dataset/adult-training.csv', names=columns)
# test = pd.read_csv('Dataset/adult-test.csv', names=columns, skiprows=1)
# train.info()
#
# #Clean the Data
#
# df = pd.concat([train, test], axis=0)
#
# df['Income'] = df['Income'].apply(lambda x: 1 if x == ' >50K' else 0)
#
#
# #REMOVE UNKNOWNS
#
# df.replace(' ?', np.nan, inplace=True)  ###making copy for visualization
#
# # Preparing data for Training and testing
#
# X = np.array(df.drop(['Income'], 1))
# y = np.array(df['Income'])
# X = preprocessing.scale(X)
# y = np.array(df['Income'])
#
# # Splitting data as train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# # Decision tree
#
# clf_tree = DecisionTreeClassifier(min_samples_leaf=3, random_state=42)
#
# clf_tree.fit(X_train, y_train)
# tree_predict = clf_tree.predict(X_test)
# metrics.accuracy_score(y_test, tree_predict)
#
# print(confusion_matrix(y_test, tree_predict))
# print(classification_report(y_test, tree_predict))
# DTA = accuracy_score(y_test, tree_predict)
# print("The Accuracy for Decision Tree Model is {}".format(DTA))

# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

column_names = ["sex", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]
# column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
# df = pd.read_csv("Dataset/iris.data", names=column_names)
df = pd.read_csv("Dataset/abalone.data", names=column_names)

df['rings'] = df['rings'].apply(lambda x: 'young' if x <= 7.5 else ('adult' if x <= 13 else 'old'))

for label in "MFI":
    df[label] = df["sex"] == label
del df["sex"]

# y = df.rings.values
# del df["rings"]
# X = df.values.astype(np.float)

# X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# y = df['Species'].values

X = df[["M", "F", "I", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight"]].values
y = df['rings'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(X, y, train_size=0.7, random_state=1)

# train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2)

# model = DecisionTreeClassifier(max_depth=4)

dtc = DecisionTreeClassifier(max_depth=4)
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))


# model.fit(train_X, train_y)
# predicted_test_y = model.predict(test_X)
#
# print(accuracy_score(test_y, predicted_test_y))
