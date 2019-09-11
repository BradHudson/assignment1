import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def main():
    df = pd.read_csv("Dataset/pendigits.csv", header=None)
    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    training_x, testing_x, training_y, testing_y = train_test_split(X, Y, test_size=0.4, random_state=seed, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(training_x)
    training_x = scaler.transform(training_x)
    testing_x = scaler.transform(testing_x)

    neuron_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    for i in range(1, 50):
        neuron_array.append(i)
        learner2 = DecisionTreeClassifier(max_depth=i + 1, random_state=seed)
        boosted_learner2 = AdaBoostClassifier(algorithm='SAMME', base_estimator=learner2, random_state=seed)
        cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=3).mean())

        boosted_learner2.fit(training_x, training_y)
        training_depth_array.append(boosted_learner2.score(training_x, training_y))
        testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))

    plt.plot(neuron_array, training_depth_array, label='Training')
    plt.plot(neuron_array, testing_depth_array, label='Testing')
    plt.plot(neuron_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Accuracy vs Max Depth")
    plt.ylabel('Accuracy %')
    plt.xlabel('Max Depth')
    plt.xlim([1, 50])
    plt.savefig('pendigitsboostedmaxdepth.png')
    plt.close()

    layers_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    for i in range(1, 50):
        layers_array.append(i)
        learner2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=([i + 1]), random_state=seed)
        cross_val_score_array.append(cross_val_score(learner2, training_x, training_y, cv=3).mean())

        learner2.fit(training_x, training_y)
        training_depth_array.append(learner2.score(training_x, training_y))
        testing_depth_array.append(learner2.score(testing_x, testing_y))

    plt.plot(layers_array, training_depth_array, label='Training')
    plt.plot(layers_array, testing_depth_array, label='Testing')
    plt.plot(layers_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Accuracy vs Estimator Count")
    plt.ylabel('Accuracy %')
    plt.xlabel('Number of Estimators')
    plt.xlim([1, 50])
    plt.savefig('pendigitsboostedestimators.png')
    plt.close()


if __name__== "__main__":
  main()
