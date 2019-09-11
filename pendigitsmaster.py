import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv("Dataset/pendigits.csv", header=None)
    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    training_x1, testing_x1, training_y, testing_y = train_test_split(X, Y, test_size=0.3, random_state=seed, shuffle=True)

    standardScalerX = StandardScaler()
    training_x = standardScalerX.fit_transform(training_x1)
    testing_x = standardScalerX.fit_transform(testing_x1)


    # # DT Max Depth Gini
    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('DT Max Depth Gini')
    # for i in range(1, 50):
    #     max_depth_array.append(i)
    #     learner = DecisionTreeClassifier(criterion='gini',max_depth=i + 1, random_state=seed)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(max_depth_array, training_depth_array, label='Training')
    # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('pendigitsmaxdepthGini.png')
    # plt.close()
    #
    # # DT Max Depth Entropy
    #
    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('DT Max Depth Entropy')
    # for i in range(1, 50):
    #     max_depth_array.append(i)
    #     learner = DecisionTreeClassifier(criterion='entropy',max_depth=i + 1, random_state=seed)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(max_depth_array, training_depth_array, label='Training')
    # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('pendigitsmaxdepthEntropy.png')
    # plt.close()

    # Adaboost Max Depth

    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('Adaboost Max Depth')
    # for i in range(1, 50):
    #     max_depth_array.append(i)
    #     learner2 = DecisionTreeClassifier(max_depth=i + 1, random_state=seed)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=learner2, random_state=seed)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(max_depth_array, training_depth_array, label='Training')
    # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('pendigitsboostedmaxdepth.png')
    # plt.close()
    #
    # # Adaboost Estimators
    #
    # estimator_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # n_estimators = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]
    #
    # print('Adaboost Estimators')
    # for i in n_estimators:
    #     estimator_array.append(i)
    #     # learner2 = DecisionTreeClassifier(max_depth=1,random_state=seed)
    #     boosted_learner2 = AdaBoostClassifier(random_state=seed,
    #                                           n_estimators=i + 1)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(estimator_array, training_depth_array, label='Training')
    # plt.plot(estimator_array, testing_depth_array, label='Testing')
    # plt.plot(estimator_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Estimator Count")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Number of Estimators')
    # plt.xlim([1, 50])
    # plt.savefig('pendigitsboostedestimators.png')
    # plt.close()

    # Adaboost Learning Rate

    # learning_rate_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # learning_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
    #
    # print('Adaboost Learning Rate')
    # for i in learning_rates:
    #     learning_rate_array.append(i)
    #     boosted_learner2 = AdaBoostClassifier(random_state=seed,
    #                                           learning_rate=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(learning_rate_array, training_depth_array, label='Training')
    # plt.plot(learning_rate_array, testing_depth_array, label='Testing')
    # plt.plot(learning_rate_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Estimator Count")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Learning Rate')
    # plt.xlim([0, 1])
    # plt.savefig('pendigitsboostedLearningRate.png')
    # plt.close()

    # KNN Number of Neighbors

    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors')
    # for i in range(1, 50, 3):
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(n_neighbors=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(knn_array, training_depth_array, label='Training')
    # plt.plot(knn_array, testing_depth_array, label='Testing')
    # plt.plot(knn_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs K Neighbors")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('K Neighbors')
    # plt.xlim([1, 50])
    # plt.savefig('pendigitsKNN.png')
    # plt.close()

    # KNN Weight

    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Weights')
    # for i in ['uniform','distance']:
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(weights=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # print('--------------------------')
    # print('KNN Uniform Weight Results')
    # print('Training Accuracy: ' + str(training_depth_array[0]))
    # print('Testing Accuracy: ' + str(testing_depth_array[0]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[0]))
    # print('--------------------------')
    # print('KNN Distance Weight Results')
    # print('Training Accuracy: ' + str(training_depth_array[1]))
    # print('Testing Accuracy: ' + str(testing_depth_array[1]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[1]))
    # print('--------------------------')

    # # KNN Metric
    #
    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Weights')
    # for i in [1,2]:
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(p=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # print('--------------------------')
    # print('KNN Manhattan Distance Results')
    # print('Training Accuracy: ' + str(training_depth_array[0]))
    # print('Testing Accuracy: ' + str(testing_depth_array[0]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[0]))
    # print('--------------------------')
    # print('KNN Euclidean Distance Results')
    # print('Training Accuracy: ' + str(training_depth_array[1]))
    # print('Testing Accuracy: ' + str(testing_depth_array[1]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[1]))
    # print('--------------------------')




if __name__== "__main__":
  main()
