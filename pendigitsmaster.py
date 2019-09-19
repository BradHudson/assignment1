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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time

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
    # plot_validation_curve(max_depth_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Max Depth Gini", 'Score', 'Max Depth', [1, 50],
    #                       'PenDigitsPlots/pendigitsmaxdepthGini.png')
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
    # plot_validation_curve(max_depth_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Max Depth Entropy", 'Score', 'Max Depth', [1, 50],
    #                       'PenDigitsPlots/pendigitsmaxdepthEntropy.png')
    #
    # # DT Random Search & Learning Curve
    # max_depths = np.arange(1, 20, 1)
    # params = {'criterion': ['gini', 'entropy'], 'max_depth': max_depths}
    # learner = DecisionTreeClassifier(random_state=seed)
    #
    # dt_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=40)
    # start = time.clock()
    # dt_cv.fit(training_x, training_y)
    # dt_train_time = time.clock() - start
    # print('Time to Train: ' + str(dt_train_time))
    # print('Training Accuracy: ' + str(dt_cv.score(training_x, training_y)))
    # print('Testing Accuracy: ' + str(dt_cv.score(testing_x, testing_y)))
    # print(dt_cv.best_params_) #entropy, max depth 11
    # start = time.clock()
    # test_y_predicted = dt_cv.predict(testing_x)
    # dt_query_time = time.clock() - start
    # print('Time to Query: ' + str(dt_query_time))
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # # plot_confusion_matrix(testing_y, test_y_predicted, classes=(0,1,2,3,4,5,6,7,8,9),
    # #                       title='Confusion matrix')
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     dt_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsDTLearningCurve.png', "Learning Curve DT")

    # Adaboost Max Depth

    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('Adaboost Max Depth')
    # for i in range(1, 20, 1):
    #     max_depth_array.append(i)
    #     learner2 = DecisionTreeClassifier(max_depth=i, random_state=seed, criterion='entropy')
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=learner2, random_state=seed,algorithm='SAMME')
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(max_depth_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Max Depth of Base Estimator", 'Score', 'Max Depth', [1, 20],
    #                       'PenDigitsPlots/pendigitsboostedmaxdepth.png')
    #
    # # Adaboost Estimators
    #
    # estimator_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # n_estimators = range(1, 105, 5)
    #
    # print('Adaboost Estimators')
    # for i in n_estimators:
    #     estimator_array.append(i)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=1),algorithm='SAMME',random_state=seed, n_estimators=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(estimator_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Estimator Count", 'Score', 'Number of Estimators', [1, 100],
    #                       'PenDigitsPlots/pendigitsboostedestimators.png')
    #
    # # Adaboost Learning Rate
    #
    # learning_rate_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # learning_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.5, 2.0]
    #
    # print('Adaboost Learning Rate')
    # for i in learning_rates:
    #     learning_rate_array.append(i)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=1),algorithm='SAMME', random_state=seed, learning_rate=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(learning_rate_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Learning Rates", 'Score', 'Learning Rate', [0, 2],
    #                       'PenDigitsPlots/pendigitsboostedLearningRate.png')

    # # Adaboost Random Search & Learning Curve
    #
    # max_depths = np.arange(1, 20, 1)
    # params = {'n_estimators': [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], 'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1, 1.5, 2.0], 'base_estimator__max_depth': max_depths}
    # learner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1),random_state=seed)
    #
    # print('starting grid  search')
    # boost_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # start = time.clock()
    # boost_cv.fit(training_x, training_y)
    # dt_train_time = time.clock() - start
    # print('Time to Train: ' + str(dt_train_time))
    # print('Training Accuracy: ' + str(boost_cv.score(training_x, training_y)))
    # print('Testing Accuracy: ' + str(boost_cv.score(testing_x, testing_y)))
    # print(boost_cv.best_params_)  # entropy, max depth 11
    # start = time.clock()
    # test_y_predicted = boost_cv.predict(testing_x)
    # dt_query_time = time.clock() - start
    # print('Time to Query: ' + str(dt_query_time))
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    #
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     boost_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsboostedLearningCurve.png', "Learning Curve Boosted DT")

    # KNN Number of Neighbors with different weights

    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors with Manhattan Distance')
    # for i in range(1, 50, 1):
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(n_neighbors=i, p=1)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(knn_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs k Neighbors Manhattan", 'Score', 'k Neighbors', [1, 50],
    #                       'PenDigitsPlots/pendigitsManhattanKNN.png')
    #
    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors with Euclidean Distance')
    # for i in range(1, 50, 1):
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(n_neighbors=i,p=2)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(knn_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs k Neighbors Euclidean", 'Score', 'k Neighbors', [1, 50],
    #                       'PenDigitsPlots/pendigitsEuclideanKNN.png')

    # params = {'p': [1, 2], 'n_neighbors': np.arange(2, 50, 1)}
    # learner = KNeighborsClassifier()
    #
    # print('starting random  search')
    # knn_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=100)
    # start = time.clock()
    # knn_cv.fit(training_x, training_y)
    # dt_train_time = time.clock() - start
    # print('Time to Train: ' + str(dt_train_time))
    # print('Training Accuracy: ' + str(knn_cv.score(training_x, training_y)))
    # print('Testing Accuracy: ' + str(knn_cv.score(testing_x, testing_y)))
    # print(knn_cv.best_params_)  # entropy, max depth 11
    # start = time.clock()
    # test_y_predicted = knn_cv.predict(testing_x)
    # dt_query_time = time.clock() - start
    # print('Time to Query: ' + str(dt_query_time))
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     knn_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsKNNLearningCurve.png', "Learning Curve kNN")

    # ANN 1 Layer with different number of neurons

    ann_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    print('ANN Number of Neurons')
    for i in [0,1,5,10,15,20,25,30,35,40,45,49]:
        print('------hey we are on ' + str(i))
        ann_array.append(i)
        learner = MLPClassifier(hidden_layer_sizes=([i+1]))
        cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
        learner.fit(training_x, training_y)
        training_depth_array.append(learner.score(training_x, training_y))

    plot_validation_curve(ann_array, training_depth_array, cross_val_score_array,
                          "Cross Validation Score vs Neurons in One Hidden Layer", 'Score', 'Number of Neurons', [1, 50],
                          'PenDigitsPlots/pendigitsANNNeurons.png')

    # ANN Neurons per Layers

    ann_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    print('ANN Number of Layers')
    for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
        print('------hey we are on ' + str(i))
        hidden_layers = []
        for x in range(i):
            hidden_layers.append(10)
        ann_array.append(i)
        learner = MLPClassifier(hidden_layer_sizes=(hidden_layers))
        cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
        learner.fit(training_x, training_y)
        training_depth_array.append(learner.score(training_x, training_y))
        #testing_depth_array.append(learner.score(testing_x, testing_y))

    plt.plot(ann_array, training_depth_array, label='Training')
    #plt.plot(ann_array, testing_depth_array, label='Testing')
    plt.plot(ann_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Accuracy vs Number of Hidden Layers")
    plt.ylabel('Accuracy %')
    plt.xlabel('Number of Hidden Layers')
    plt.xlim([1, 50])
    plt.savefig('PenDigitsPlots/pendigitsANNLayers.png')
    plt.close()

    plot_validation_curve(ann_array, training_depth_array, cross_val_score_array,
                          "Cross Validation Score vs # of Hidden Layers", 'Score', 'Number of Hidden Layers', [1, 50],
                          'PenDigitsPlots/pendigitsANNLayers.png')

    params = {'hidden_layer_sizes': [(16,16), (8,8), (16,), (8,)],
              'alpha': np.arange(0.0001, .01, 0.0005), 'activation': ['relu', 'logistic']}
    learner = MLPClassifier()

    print('starting grid  search')
    ann_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    ann_cv.fit(training_x, training_y)
    print(ann_cv.score(testing_x, testing_y))
    print(ann_cv.best_params_)
    test_y_predicted = ann_cv.predict(testing_x)
    y_true = pd.Series(testing_y)
    y_pred = pd.Series(test_y_predicted)
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    train_sizes, train_scores, test_scores = learning_curve(
        ann_cv,
        training_x,
        training_y, n_jobs=-1,
        cv=2,
        train_sizes=np.linspace(.1, 1.0, 10),
        random_state=seed)

    plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsANNLearningCurve.png')

    # ANN over max iterations

    ann_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    print('ANN Different Epochs')
    for i in [200, 400, 600, 800, 1000, 1500, 2000]:
        print('------hey we are on ' + str(i))
        ann_array.append(i)
        learner = MLPClassifier(hidden_layer_sizes=(16,16), alpha=0.0041, activation='relu') # best params from previous random search
        cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
        learner.fit(training_x, training_y)
        training_depth_array.append(learner.score(training_x, training_y))
        #testing_depth_array.append(learner.score(testing_x, testing_y))

    plt.plot(ann_array, training_depth_array, label='Training')
    #plt.plot(ann_array, testing_depth_array, label='Testing')
    plt.plot(ann_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Cross Validation Score vs. Max Iterations")
    plt.ylabel('Score')
    plt.xlabel('Max Number of Iterations')
    plt.xlim([0, 2000])
    plt.savefig('PenDigitsPlots/pendigitsANNMaxIterations.png')
    plt.close()

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

    # SVM Kernels Sigmoid vs RBF

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels Sigmoid Different Gamma Values')
    # for i in np.arange(0.01, 1, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='sigmoid', gamma=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     #testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(svm_array, training_depth_array, label='Training')
    # #plt.plot(ann_array, testing_depth_array, label='Testing')
    # plt.plot(svm_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Cross Validation Score vs. Gamma Values - Sigmoid Kernel")
    # plt.ylabel('Score')
    # plt.xlabel('Gamma Values')
    # plt.xlim([0.00, 1.0])
    # plt.savefig('PenDigitsPlots/pendigitsGammaSigmoid.png')
    # plt.close()
    #
    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels RBF Different Gamma Values')
    # for i in np.arange(0.01, 1, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='rbf', gamma=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     # testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(svm_array, training_depth_array, label='Training')
    # # plt.plot(ann_array, testing_depth_array, label='Testing')
    # plt.plot(svm_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Cross Validation Score vs. Gamma Values - RBF Kernel")
    # plt.ylabel('Score')
    # plt.xlabel('Gamma Values')
    # plt.xlim([0.00, 1.0])
    # plt.savefig('PenDigitsPlots/pendigitsGammaRBF.png')
    # plt.close()
    #
    # params = {'kernel': ['sigmoid', 'rbf'],
    #           'gamma': np.arange(0.01, 1, 0.1)}
    # learner = svm.SVC()
    #
    # print('starting grid  search')
    # svc_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # svc_cv.fit(training_x, training_y)
    # print(svc_cv.score(testing_x, testing_y))
    # print(svc_cv.best_params_)
    # test_y_predicted = svc_cv.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     svc_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'PenDigitsPlots/pendigitsSVCLearningCurve.png')

    # SVM over Epochs

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('ANN Different Epochs')
    # for i in [1, 10, 50, 75, 150, 200, 400]:
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='rbf', gamma=0.21, max_iter=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #     #testing_depth_array.append(learner.score(testing_x, testing_y))
    #
    # plt.plot(svm_array, training_depth_array, label='Training')
    # #plt.plot(ann_array, testing_depth_array, label='Testing')
    # plt.plot(svm_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Cross Validation Score vs. Max Iterations", fontsize=30)
    # plt.ylabel('Score')
    # plt.xlabel('Max Number of Iterations')
    # plt.xlim([0, 400])
    # plt.savefig('PenDigitsPlots/pendigitsSVMMaxIterations.png')
    # plt.close()

def plot_validation_curve(param_array,training_array,cross_val_array,title,y, x, limit,file):
    plt.plot(param_array, training_array, label='Training')
    # plt.plot(max_depth_array, testing_depth_array, label='Testing')
    plt.plot(param_array, cross_val_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title(title, fontdict={'size': 16})
    plt.ylabel(y)
    plt.xlabel(x)
    plt.xlim(limit)
    plt.savefig(file)
    plt.close()


def plot_learning_curve(train_scores, test_scores, train_sizes, file_name, title=""):
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
    plt.title(title, fontdict={'size': 16})

    plt.savefig(file_name)
    plt.close()


if __name__== "__main__":
  main()
