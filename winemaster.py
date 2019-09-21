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
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import validation_curve
import time

def main():
    df = pd.read_csv("Dataset/winequality-white.csv", delimiter=";")

    n, bins, patches = plt.hist(x=np.array(df.iloc[:, -1]), bins='auto', color='#0504aa',
                                alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
    plt.title('Class Distribution')
    plt.ylim(ymax=2200)
    plt.savefig('WinePlots/wineClassDistributionOriginal.png')
    plt.close()

    lowquality = df.loc[df['quality'] <= 6].index
    highquality = df.loc[df['quality'] > 6].index
    df.iloc[lowquality, df.columns.get_loc('quality')] = 0
    df.iloc[highquality, df.columns.get_loc('quality')] = 1

    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    n, bins, patches = plt.hist(x=Y, bins='auto' ,color='#0504aa',
                                alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.xticks(np.arange(2), ('Low', 'High'))
    plt.title('Class Distribution')
    plt.ylim(ymax=4000)
    plt.savefig('WinePlots/wineClassDistribution.png')
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
    #                       'WinePlots/winemaxdepthGini.png')
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
    # plot_validation_curve(max_depth_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Max Depth Entropy", 'Score', 'Max Depth', [1, 50],
    #                       'WinePlots/winemaxdepthEntropy.png')
    #
    # DT Random Search & Learning Curve
    max_depths = np.arange(1, 20, 1)
    params = {'criterion': ['gini', 'entropy'], 'max_depth': max_depths}
    learner = DecisionTreeClassifier(random_state=seed)
    start = time.clock()
    dt_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=40)
    dt_cv.fit(training_x, training_y)
    print(dt_cv.best_params_)
    dt_train_time = time.clock() - start
    print('Time to Train: ' + str(dt_train_time))
    print('Training Accuracy: ' + str(dt_cv.score(training_x, training_y)))
    print('Testing Accuracy: ' + str(dt_cv.score(testing_x, testing_y)))
    print(dt_cv.best_params_)  # entropy, max depth 11
    start = time.clock()
    test_y_predicted = dt_cv.predict(testing_x)
    dt_query_time = time.clock() - start
    print('Time to Query: ' + str(dt_query_time))
    y_true = pd.Series(testing_y)
    y_pred = pd.Series(test_y_predicted)
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    train_sizes, train_scores, test_scores = learning_curve(
        dt_cv,
        training_x,
        training_y, n_jobs=-1,
        cv=10,
        train_sizes=np.linspace(.1, 1.0, 10),
        random_state=seed)

    plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineDTLearningCurve.png', "Learning Curve DT")

    # # Adaboost Max Depth
    #
    # max_depth_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('Adaboost Max Depth')
    # for i in range(1, 20, 2):
    #     max_depth_array.append(i)
    #     learner2 = DecisionTreeClassifier(max_depth=i, criterion='gini', random_state=seed)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=learner2, random_state=seed, algorithm='SAMME')
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(max_depth_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Max Depth of Base Estimator", 'Score', 'Max Depth', [1, 20],
    #                       'WinePlots/wineboostedmaxdepth.png')
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
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',max_depth=1),algorithm='SAMME',random_state=seed, n_estimators=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(estimator_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Number of Estimators", 'Score', 'Number of Estimators', [1, 50],
    #                       'WinePlots/wineboostedestimators.png')
    #
    # Adaboost Learning Rate

    # learning_rate_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    # learning_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1, 1.5, 2]
    #
    # print('Adaboost Learning Rate')
    # for i in learning_rates:
    #     learning_rate_array.append(i)
    #     boosted_learner2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',max_depth=1),algorithm='SAMME', random_state=seed, learning_rate=i)
    #     cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=10).mean())
    #
    #     boosted_learner2.fit(training_x, training_y)
    #     training_depth_array.append(boosted_learner2.score(training_x, training_y))
    #
    # plot_validation_curve(learning_rate_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs Learning Rates", 'Score', 'Learning Rate', [0, 2],
    #                       'WinePlots/wineboostedLearningRate.png')

    # # Adaboost Random Search & Learning Curve
    #
    # max_depths = np.arange(1, 20, 1)
    # params = {'n_estimators': [10, 15, 20, 25, 30, 40, 50], 'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1], 'base_estimator__max_depth': max_depths}
    # learner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'),random_state=seed)
    #
    # print('starting grid  search')
    # boost_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # start = time.clock()
    # boost_cv.fit(training_x, training_y)
    # dt_train_time = time.clock() - start
    # print('Time to Train: ' + str(dt_train_time))
    # print('Training Accuracy: ' + str(boost_cv.score(training_x, training_y)))
    # print('Testing Accuracy: ' + str(boost_cv.score(testing_x, testing_y)))
    # print(boost_cv.best_params_)
    # start = time.clock()
    # test_y_predicted = boost_cv.predict(testing_x)
    # dt_query_time = time.clock() - start
    # print('Time to Query: ' + str(dt_query_time))
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     boost_cv,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineboostedLearningCurve.png')

    # KNN Number of Neighbors

    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors with Manhattan Distance')
    # for i in range(1, 50, 2):
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(n_neighbors=i,p=1)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(knn_array, training_depth_array, cross_val_score_array,
    #                                             "Cross Validation Score vs k Neighbors Manhattan", 'Score', 'k Neighbors', [1, 50],
    #                                             'WinePlots/wineManhattanKNN.png')
    #
    # knn_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('KNN Number of Neighbors with Euclidean Distance')
    # for i in range(1, 50, 2):
    #     knn_array.append(i)
    #     learner = KNeighborsClassifier(n_neighbors=i, p=2)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(knn_array, training_depth_array, cross_val_score_array,
    #                                             "Cross Validation Score vs k Neighbors Euclidean", 'Score', 'k Neighbors', [1, 50],
    #                                             'WinePlots/wineEuclideanKNN.png')
    #
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
    # print(knn_cv.best_params_)
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
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineKNNLearningCurve.png')

    # ANN 1 Layer with different number of neurons

    # ann_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('ANN Number of Neurons')
    # for i in [1,5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100]:
    #     print('------hey we are on ' + str(i))
    #     ann_array.append(i)
    #     learner = MLPClassifier(hidden_layer_sizes=([i]))
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(ann_array, training_depth_array, cross_val_score_array,
    #                                             "Cross Validation Score vs Neurons in One Hidden Layer", 'Score', 'Number of Neurons', [1, 100],
    #                                             'WinePlots/wineANNNeurons.png')

    # # ANN Neurons per Layers

    # ann_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('ANN Number of Layers')
    # for i in [1, 3, 5, 8, 10, 11, 13, 15, 17, 20, 23, 25]:
    #     print('------hey we are on ' + str(i))
    #     hidden_layers = []
    #     for x in range(i):
    #         hidden_layers.append(22)
    #     ann_array.append(i)
    #     learner = MLPClassifier(hidden_layer_sizes=(hidden_layers), activation='relu', alpha=0.0051)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(ann_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs # of Hidden Layers", 'Score', 'Number of Hidden Layers', [1, 25],
    #                       'WinePlots/wineANNLayers.png')

    # ANN Learning Curve

    # params = {'hidden_layer_sizes': [(11,11), (5,5), (11,), (5,), (22,), (22, 22), (5, 5, 5), (11, 11, 11), (22, 22, 22)], 'alpha': np.arange(0.0001, 0.01, 0.005), 'activation': ['relu', 'logistic']}
    # learner = MLPClassifier(max_iter=500, random_state=seed)
    # ##### best params {'hidden_layer_sizes': (11,11), 'alpha': 0.0001, 'activation': 'relu'}
    # # ann_cv = MLPClassifier(max_iter=3000,hidden_layer_sizes=(22,22,22), alpha=0.0051, activation='relu', random_state=seed)
    # print('starting random  search')
    # ann_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=20, verbose=1000)
    # ann_cv.fit(training_x, training_y)
    # print(ann_cv.best_params_)
    #
    # final_ann = MLPClassifier(**ann_cv.best_params_)
    #
    # start = time.clock()
    # final_ann.fit(training_x, training_y)
    #
    # dt_train_time = time.clock() - start
    # # print('refit time: ' + str(final_ann.refit_time_))
    # # print(final_ann.best_params_)
    # print('Time to Train: ' + str(dt_train_time))
    # print('Training Accuracy: ' + str(final_ann.score(training_x, training_y)))
    # print('Testing Accuracy: ' + str(final_ann.score(testing_x, testing_y)))
    # # print(final_ann.best_params_)
    # start = time.clock()
    # test_y_predicted = final_ann.predict(testing_x)
    # dt_query_time = time.clock() - start
    # print('Time to Query: ' + str(dt_query_time))
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     final_ann,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=5,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineANNLearningCurve.png')


    #ANN over epochs

    # ann_array = []
    # training_depth_array = []
    # cross_val_score_array = []
    # testing_depth_array = []
    #
    # learner = MLPClassifier(hidden_layer_sizes=(22,), alpha=0.0051, activation='relu', max_iter=1,
    #                         random_state=seed, verbose=10, warm_start=True)
    # for i in np.arange(3000):
    #     ann_array.append(i)
    #     learner = learner.fit(training_x,training_y)
    #     score = learner.score(training_x, training_y)
    #     print(score)
    #     training_depth_array.append(score)
    #     cross_score = learner.score(testing_x, testing_y)
    #     cross_val_score_array.append(cross_score)
    #     print(cross_score)
    #
    #
    #
    # plot_validation_curve(ann_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs. Epochs", 'Score', 'Epochs', [0, 3000],
    #                       'WinePlots/wineANNEpochs.png')

    # SVM Kernels Sigmoid vs RBF

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels Sigmoid Different Gamma Values')
    # for i in np.arange(0.01, 2, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='sigmoid', gamma=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plt.plot(svm_array, training_depth_array, label='Training')
    # plt.plot(svm_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Cross Validation Score vs. Gamma Values - Sigmoid Kernel")
    # plt.ylabel('Score')
    # plt.xlabel('Gamma Values')
    # plt.xlim([0.00, 2.0])
    # plt.savefig('WinePlots/wineGammaSigmoid.png')
    # plt.close()

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels RBF Different Gamma Values')
    # for i in np.arange(0.01, 2, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='rbf', gamma=i)
    #     cross_score = cross_val_score(learner, training_x, training_y, cv=10).mean()
    #     print(cross_score)
    #     cross_val_score_array.append(cross_score)
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plt.plot(svm_array, training_depth_array, label='Training')
    # plt.plot(svm_array, cross_val_score_array, label='Cross Validation')
    # plt.legend(loc=4, fontsize=8)
    # plt.title("Cross Validation Score vs. Gamma Values - RBF Kernel")
    # plt.ylabel('Score')
    # plt.xlabel('Gamma Values')
    # plt.xlim([0.00, 2.0])
    # plt.savefig('WinePlots/wineGammaRBF.png')
    # plt.close()

    # # SVM C Values
    #
    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels Sigmoid Different C Values')
    # for i in np.arange(0.01, 2, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='sigmoid', C=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(svm_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs. C Values - Sigmoid Kernel", 'Score', 'C Values', [0.00, 2.0],
    #                       'WinePlots/wineCSigmoid.png')
    #
    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Kernels RBF Different C Values')
    # for i in np.arange(0.01, 2, 0.1):
    #     print('------hey we are on ' + str(i))
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='rbf', C=i)
    #     cross_val_score_array.append(cross_val_score(learner, training_x, training_y, cv=10).mean())
    #     learner.fit(training_x, training_y)
    #     training_depth_array.append(learner.score(training_x, training_y))
    #
    # plot_validation_curve(svm_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs. C Values - RBF Kernel", 'Score', 'C Values', [0.00, 2.0],
    #                       'WinePlots/wineCRBF.png')

    # #Learning Curve Sigmoid
    #
    # params = {'gamma': np.arange(0.01, 2, 0.1), 'C':np.arange(0.01, 1, 0.1)}
    # learner = svm.SVC(kernel='sigmoid')
    #
    # print('starting grid  search')
    # svc_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # svc_cv.fit(training_x, training_y)
    # best_params = svc_cv.best_params_ #{'gamma': 0.51, 'C': 0.01}
    # final_svc = svm.SVC(kernel='sigmoid', **best_params)
    # final_svc.fit(training_x, training_y)
    # print(final_svc.score(testing_x, testing_y))
    # print(final_svc.score(training_x, training_y))
    # test_y_predicted = final_svc.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     final_svc,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineSVCLearningCurveSigmoid.png')
    #
    # # Learning Curve RBF
    #
    # params = {'gamma': np.arange(0.01, 2, 0.1), 'C':np.arange(0.01, 1, 0.1)}
    # learner = svm.SVC(kernel='rbf')
    #
    # print('starting grid  search')
    # svc_cv = RandomizedSearchCV(learner, n_jobs=1, param_distributions=params, refit=True, n_iter=50)
    # svc_cv.fit(training_x, training_y)
    # best_params = svc_cv.best_params_ #{'gamma': 1.31, 'C': 0.91}
    # final_svc = svm.SVC(kernel='rbf', **best_params)
    # final_svc.fit(training_x, training_y)
    # print(final_svc.score(testing_x, testing_y))
    # print(final_svc.score(training_x, training_y))
    # test_y_predicted = final_svc.predict(testing_x)
    # y_true = pd.Series(testing_y)
    # y_pred = pd.Series(test_y_predicted)
    # print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #
    # train_sizes, train_scores, test_scores = learning_curve(
    #     final_svc,
    #     training_x,
    #     training_y, n_jobs=-1,
    #     cv=10,
    #     train_sizes=np.linspace(.1, 1.0, 10),
    #     random_state=seed)
    #
    # plot_learning_curve(train_scores, test_scores, train_sizes, 'WinePlots/wineSVCLearningCurveRBF.png')

    # SVM over Epochs Sigmoid

    # svm_array = []
    # training_depth_array = []
    # testing_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Different Epochs Sigmoid')
    # for i in np.arange(1000):
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='sigmoid', verbose=100, max_iter=i)
    #     learner = learner.fit(training_x,training_y)
    #     score = learner.score(training_x, training_y)
    #     print(score)
    #     training_depth_array.append(score)
    #     cross_score = learner.score(testing_x, testing_y)
    #     cross_val_score_array.append(cross_score)
    #
    # plot_validation_curve(svm_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs. Epochs", 'Score', 'Epochs', [0, 1000],
    #                       'WinePlots/wineSVMEpochsSigmoid.png')

    # # SVM over Epochs RBF
    # svm_array = []
    # training_depth_array = []
    # cross_val_score_array = []
    #
    # print('SVM Different Epochs RBF')
    # for i in np.arange(1000):
    #     svm_array.append(i)
    #     learner = svm.SVC(kernel='rbf', verbose=100, max_iter=i)
    #     learner = learner.fit(training_x, training_y)
    #     score = learner.score(training_x, training_y)
    #     print(score)
    #     training_depth_array.append(score)
    #     cross_score = learner.score(testing_x, testing_y)
    #     cross_val_score_array.append(cross_score)
    #
    # plot_validation_curve(svm_array, training_depth_array, cross_val_score_array,
    #                       "Cross Validation Score vs. Epochs", 'Score', 'Epochs', [0, 1000],
    #                       'WinePlots/wineSVMEpochsRBF.png')

    # Timing Wine

    # # Training Time
    # dt_clf = DecisionTreeClassifier(max_depth=19, criterion='gini')
    # ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=11),n_estimators=40, learning_rate=1)
    # knn_clf = KNeighborsClassifier(p=1, n_neighbors=2)
    # ann_clf = MLPClassifier(hidden_layer_sizes=(22,), alpha=0.0051, activation='relu')
    # svm_rbf_clf = svm.SVC(kernel='rbf', gamma=1.31, C=0.91)
    # svm_sigmoid_clf = svm.SVC(kernel='sigmoid', gamma=0.51, C=0.01)
    # labels = ["Decision Tree", "Adaboost", "KNN", "ANN", "SVM_RBF", "SVM_Sigmoid"]
    # count = 0
    # for clf in [dt_clf,ada_clf,knn_clf, ann_clf, svm_rbf_clf, svm_sigmoid_clf]:
    #     iteration_array = []
    #     train_array = []
    #     query_array = []
    #     for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         if count == 3:
    #             clf = MLPClassifier(hidden_layer_sizes=(22,), alpha=0.0051, activation='relu')
    #         if count == 4:
    #             clf = svm.SVC(kernel='rbf', gamma=1.31, C=0.91)
    #         if count == 5:
    #             clf = svm.SVC(kernel='sigmoid', gamma=0.51, C=0.01)
    #         X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=seed)
    #         iteration_array.append(X_train.shape[0])
    #         st = time.clock()
    #         clf.fit(X_train, y_train)
    #         train_time = time.clock() - st
    #         train_array.append(train_time)
    #         # st = time.clock()
    #         # clf.predict(X_test)
    #         # query_time = time.clock() - st
    #         # query_array.append(query_time)
    #     plt.plot(iteration_array, train_array, label=labels[count])
    #     # plt.plot(iteration_array, query_array, label=str(clf) + 'Query Time')
    #     plt.legend(loc=4, fontsize=8)
    #     plt.title("Training Times for Learners", fontdict={'size': 16})
    #     plt.ylabel("Time")
    #     plt.xlabel("Iteration Size")
    #     count = count + 1
    # plt.savefig("WineTrainingTimes.png")
    # plt.close()
    #
    # # Query Time
    #
    # dt_clf = DecisionTreeClassifier(max_depth=19, criterion='gini')
    # ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=11),n_estimators=40, learning_rate=1)
    # knn_clf = KNeighborsClassifier(p=1, n_neighbors=2)
    # ann_clf = MLPClassifier(hidden_layer_sizes=(22,), alpha=0.0051, activation='relu')
    # svm_rbf_clf = svm.SVC(kernel='rbf', gamma=1.31, C=0.91)
    # svm_sigmoid_clf = svm.SVC(kernel='sigmoid', gamma=0.51, C=0.01)
    # labels = ["Decision Tree", "Adaboost", "KNN", "ANN", "SVM_RBF", "SVM_Sigmoid"]
    # count = 0
    # for clf in [dt_clf, ada_clf, knn_clf, ann_clf, svm_rbf_clf, svm_sigmoid_clf]:
    #     iteration_array = []
    #     train_array = []
    #     query_array = []
    #     for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         if count == 3:
    #             clf = MLPClassifier(hidden_layer_sizes=(22,), alpha=0.0051, activation='relu')
    #         if count == 4:
    #             clf = svm.SVC(kernel='rbf', gamma=1.31, C=0.91)
    #         if count == 5:
    #             clf = svm.SVC(kernel='sigmoid', gamma=0.51, C=0.01)
    #         X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=seed)
    #         iteration_array.append(X_train.shape[0])
    #         st = time.clock()
    #         clf.fit(X_train, y_train)
    #         train_time = time.clock() - st
    #         train_array.append(train_time)
    #         st = time.clock()
    #         clf.predict(X_test)
    #         query_time = time.clock() - st
    #         query_array.append(query_time)
    #     #plt.plot(iteration_array, train_array, label=labels[count])
    #     plt.plot(iteration_array, query_array, label=labels[count])
    #     plt.legend(loc=4, fontsize=8)
    #     plt.title("Query Times for Learners", fontdict={'size': 16})
    #     plt.ylabel("Time")
    #     plt.xlabel("Iteration Size")
    #     count = count + 1
    # plt.savefig("WineQueryTimes.png")
    # plt.close()


def plot_validation_curve(param_array,training_array,cross_val_array,title,y, x, limit,file, testing_depth_array = []):
    plt.plot(param_array, training_array, label='Training')
    if len(testing_depth_array) > 0:
        plt.plot(param_array, testing_depth_array, label='Testing')
    if len(cross_val_array) > 0:
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
