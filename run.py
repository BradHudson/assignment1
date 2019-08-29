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
df = pd.read_csv("Dataset/abalone.data", names=column_names)

df['rings'] = df['rings'].apply(lambda x: 'young' if x <= 7.5 else ('adult' if x <= 13 else 'old'))

for label in "MFI":
    df[label] = df["sex"] == label
del df["sex"]

X = df[["M", "F", "I", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight"]].values
y = df['rings'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(X, y, train_size=0.8, random_state=40)

dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, criterion='entropy')
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))
