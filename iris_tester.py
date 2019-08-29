import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
df = pd.read_csv("Dataset/iris.data", names=column_names)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(X, y, train_size=0.7, random_state=1)

dtc = DecisionTreeClassifier(max_depth=4)
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))
