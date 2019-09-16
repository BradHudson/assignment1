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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import validation_curve
from sklearn import svm
import seaborn as sns

def main():
    df = pd.read_csv("Dataset/pendigits.csv", header=None)
    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    n, bins, patches = plt.hist(x=Y, bins='auto', color='#0504aa',
                                alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
    plt.title('Class Distribution')
    plt.ylim(ymax=1200)
    plt.savefig('PenDigitsPlots/pendigitsClassDistribution.png')
    plt.close()

    training_x1, testing_x1, training_y, testing_y = train_test_split(X, Y, test_size=0.3, random_state=seed, shuffle=True, stratify=Y)

    standardScalerX = StandardScaler()
    training_x = standardScalerX.fit_transform(training_x1)
    testing_x = standardScalerX.fit_transform(testing_x1)


    # DT Max Depth Gini
    max_depth_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    # print('DT Max Depth Gini')
    # for i in range(1, 50):
    #     max_depth_array.append(i)
    #     learner = DecisionTreeClassifier(criterion='gini',max_depth=i + 1, random_state=seed)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))


    # plt.plot(max_depth_array, training_depth_array, label='Training')
    # # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth Gini")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsmaxdepthGini.png')
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
    # # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth Entropy")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsmaxdepthEntropy.png')
    # plt.close()
    #
    # # DT Random Search & Learning Curve
    # max_depths = np.arange(1, 20, 1)
    # params = {'criterion': ['gini', 'entropy'], 'max_depth': max_depths}
    # learner = DecisionTreeClassifier(random_state=seed)
    #
    # dt_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=40)
    # dt_cv.fit(training_x, training_y)
    # print(dt_cv.score(testing_x, testing_y))
    # print(dt_cv.best_params_) #entropy, max depth 11
    # #start timer
    # test_y_predicted = dt_cv.predict(testing_x)
    # #end timer
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # #final_dt = DecisionTreeClassifier(criterion='entropy',max_depth=11, random_state=seed)
    # train_sizes, train_scores, test_scores = learning_curve(
    #     dt_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsDTLearningCurve.png')

    # Adaboost Max Depth

    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('Adaboost Max Depth')
    # for i in range(1, 50, 2):
    #     max_depth_array.append(i)
    #     learner2 = DecisionTreeClassifier(max_depth=i, random_state=seed)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=learner2, random_state=seed)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     #testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(max_depth_array, training_depth_array, label='Training')
    # #plt.plot(max_depth_array, testing_depth_array, label='Testing')
    # plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Max Depth")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Max Depth')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsboostedmaxdepth.png')
    # plt.close()
    #
    # # Adaboost Estimators
    #
    # estimator_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # n_estimators = range(1, 55, 5)
    #
    # print('Adaboost Estimators')
    # for i in n_estimators:
    #     estimator_array.append(i)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=11),algorithm='SAMME',random_state=seed, n_estimators=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     #testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(estimator_array, training_depth_array, label='Training')
    # #plt.plot(estimator_array, testing_depth_array, label='Testing')
    # plt.plot(estimator_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Estimator Count")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Number of Estimators')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsboostedestimators.png')
    # plt.close()
    #
    # # Adaboost Learning Rate
    #
    # learning_rate_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # learning_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
    #
    # print('Adaboost Learning Rate')
    # for i in learning_rates:
    #     learning_rate_array.append(i)
    #     boosted_learner2 = boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=11),algorithm='SAMME', random_state=seed, learning_rate=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #     #testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))
    #
    # plt.plot(learning_rate_array, training_depth_array, label='Training')
    # #plt.plot(learning_rate_array, testing_depth_array, label='Testing')
    # plt.plot(learning_rate_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Learning Rates")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Learning Rate')
    # plt.xlim([0, 1])
    # plt.savefig('PenDigitsPlots/pendigitsboostedLearningRate.png')
    # plt.close()
    #
    # # Adaboost Random Search & Learning Curve
    #
    # max_depths = np.arange(1, 20, 1)
    # #params = {'criterion': ['gini', 'entropy'], 'max_depth': max_depths}
    # params = {'n_estimators': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50], 'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1], 'base_estimator__max_depth': max_depths}
    # learner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),random_state=seed)
    #
    # print('starting grid  search')
    # boost_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=40)
    # boost_cv.fit(training_x, training_y)
    # print(boost_cv.score(testing_x, testing_y))
    # print(boost_cv.best_params_) #{'n_estimators': 5, 'learning_rate': 0.64, 'base_estimator__max_depth': 9}
    # test_y_predicted = boost_cv.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # #final_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11))
    # train_sizes, train_scores, test_scores = learning_curve(
    #     boost_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsboostedLearningCurve.png')

    # KNN Number of Neighbors

    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors')
    # for i in range(1, 50, 2):
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
    # plt.savefig('PenDigitsPlots/pendigitsKNN.png')
    # plt.close()
    #
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
    #
    # # KNN Metric
    #
    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Metrics')
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

    # max_depths = np.arange(1, 50, 1)
    # params = {'p': [1, 2],
    #           'weights': ['uniform','distance'], 'n_neighbors': np.arange(1, 50, 2)}
    # learner = KNeighborsClassifier()
    #
    # print('starting grid  search')
    # knn_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=30)
    # knn_cv.fit(training_x, training_y)
    # print(knn_cv.score(testing_x, testing_y))
    # print(knn_cv.best_params_)  # {'weights': 'uniform', 'p': 2, 'n_neighbors': 1}
    # test_y_predicted = knn_cv.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # # final_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11))
    # train_sizes, train_scores, test_scores = learning_curve(
    #     knn_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsKNNLearningCurve.png')

    # ANN 1 Layer with different number of neurons

    # ann_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('ANN Number of Neurons')
    # for i in [0,1,5,10,15,20,25,30,35,40,45,49]:
    #     print('------hey we are on ' + str(i))
    #     ann_array.append(i)
    #     learner = MLPClassifier(hidden_layer_sizes=([i+1]))
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(ann_array, training_depth_array, label='Training')
    # plt.plot(ann_array, testing_depth_array, label='Testing')
    # plt.plot(ann_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Number of Neurons in One Hidden Layer")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Number of Neurons')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsANNNeurons.png')
    # plt.close()

    # ANN Neurons per Layers

    # ann_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('ANN Number of Layers')
    # for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
    #     print('------hey we are on ' + str(i))
    #     hidden_layers = []
    #     for x in range(i):
    #         hidden_layers.append(10)
    #     ann_array.append(i)
    #     learner = MLPClassifier(hidden_layer_sizes=(hidden_layers))
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(ann_array, training_depth_array, label='Training')
    # plt.plot(ann_array, testing_depth_array, label='Testing')
    # plt.plot(ann_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Accuracy vs Number of Hidden Layers")
    # plt.ylabel('Accuracy %')
    # plt.xlabel('Number of Hidden Layers')
    # plt.xlim([1, 50])
    # plt.savefig('PenDigitsPlots/pendigitsANNLayers.png')
    # plt.close()

    # max_depths = np.arange(1, 50, 1)
    # params = {'hidden_layer_sizes': [(16,16), (8,8), (16,), (8,)],
    #           'alpha': np.arange(0.0, 10.0, 0.5), 'activation': ['relu', 'logistic']}
    # learner = MLPClassifier()
    #
    # print('starting grid  search')
    # ann_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # ann_cv.fit(training_x, training_y)
    # print(ann_cv.score(testing_x, testing_y))
    # print(ann_cv.best_params_)  # {'hidden_layer_sizes': (16, 16), 'alpha': 0.5, 'activation': 'relu'}
    # test_y_predicted = ann_cv.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # # final_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11))
    # train_sizes, train_scores, test_scores = learning_curve(
    #     ann_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=2,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsANNLearningCurve.png')

    # SVM Kernels Sigmoid vs RBF

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels')
    # for i in ['sigmoid','rbf']:
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # print('--------------------------')
    # print('SVM Sigmoid Results')
    # print('Training Accuracy: ' + str(training_depth_array[0]))
    # print('Testing Accuracy: ' + str(testing_depth_array[0]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[0]))
    # print('--------------------------')
    # print('SVM RBF Results')
    # print('Training Accuracy: ' + str(training_depth_array[1]))
    # print('Testing Accuracy: ' + str(testing_depth_array[1]))
    # print('Cross Validation Accuracy: ' + str(cross_val_score_array[1]))
    # print('--------------------------')

    # params = {'kernel': ['sigmoid', 'rbf'],
    #           'gamma': ['auto', 'scale']}
    # learner = svm.SVC()
    #
    # print('starting grid  search')
    # svc_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # svc_cv.fit(training_x, training_y)
    # print(svc_cv.score(testing_x, testing_y))
    # print(svc_cv.best_params_)  # {'hidden_layer_sizes': (16, 16), 'alpha': 0.5, 'activation': 'relu'}
    # test_y_predicted = svc_cv.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # # final_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11))
    # train_sizes, train_scores, test_scores = learning_curve(
    #     svc_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsSVCLearningCurve.png')


def plot_learning_curve(train_scores, test_scores, train_sizes, file_name):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
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

    plt.savefig(file_name)
    plt.close()


if __name__== "__main__":
  main()
