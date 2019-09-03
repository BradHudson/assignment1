import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Abalone

column_names = ["sex", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]
df = pd.read_csv("Dataset/abalone.data", names=column_names)

# df['rings'] = df['rings'].apply(lambda x: 'young' if x <= 7.5 else ('adult' if x <= 13 else 'old'))

for label in "MFI":
    df[label] = df["sex"] == label
del df["sex"]


#Decision Tree

X = df[["M", "F", "I", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight"]].values
y = df['rings'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(X, y, train_size=0.8, random_state=40)

dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, criterion='entropy')
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))

#learing curve

title = "Learrning Curve Abalone"
estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, criterion='entropy')


# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

ylim =  (0.7, 1.01)
# cv=cv

title = "Learning Curves Decision Tree Classifier"
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
n_jobs = 4
# train_sizes=np.linspace(.1, 1.0, 5)

plt.figure()
plt.title(title)
if ylim is not None:
    plt.ylim(*ylim)
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores = learning_curve(estimator, train_inputs, train_classes, cv=10, n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")

plt.show()


# #wine
#
# df = pd.read_csv('winequality-white.csv',sep=';',quotechar='"')
