import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import validation_curve


def main():
    #df = pd.read_csv("Dataset/pendigits.csv", header=None)
    df = pd.read_csv("Dataset/winedata.csv", delimiter=";")

    indeksDaarlig = df.loc[df['quality'] <= 6].index
    indeksGod = df.loc[df['quality'] > 6].index
    df.iloc[indeksDaarlig, df.columns.get_loc('quality')] = 0
    df.iloc[indeksGod, df.columns.get_loc('quality')] = 1

    timings = {}
    use_cv = False
    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    training_x, testing_x, training_y, testing_y = train_test_split(X, Y, test_size=0.4, random_state=seed, shuffle=True)

    t = datetime.now()

    max_depths = np.arange(1, 50, 1)
    params = {'criterion': ['gini','entropy'], 'max_depth': max_depths}

    #learner = DecisionTreeClassifier(criterion='entropy',max_depth=9,random_state=seed)
    learner = DecisionTreeClassifier(max_depth=2,random_state=seed)

    if use_cv:
        cv = GridSearchCV(learner, n_jobs=1, param_grid=params, refit=True, cv=5)
        cv.fit(training_x, training_y)
        print(cv.score(testing_x, testing_y))
        print(cv.best_params_)
        test_y_predicted = cv.predict(testing_x)
        y_true = pd.Series(testing_y)
        y_pred = pd.Series(test_y_predicted)
        print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        train_sizes, train_scores, test_scores = learning_curve(
            cv,
            training_x,
            training_y, n_jobs=-1,
            cv=10,
            train_sizes=np.linspace(.1, 1.0, 5),
            random_state=seed)

        # v_train_scores, v_test_scores = validation_curve(cv,
        #                                                  training_x,
        #                                                  training_y,
        #                                                  param_name="max_depth",
        #                                                  param_range=max_depths,
        #                                                  cv=10,
        #                                                  scoring="accuracy",
        #                                                  n_jobs=-1)
    else:
        learner.fit(training_x, training_y)
        print('Training Score: ' + str(learner.score(training_x,training_y)))
        print('Testing Score: ' + str(learner.score(testing_x, testing_y)))
        test_y_predicted = learner.predict(testing_x)
        y_true = pd.Series(testing_y)
        y_pred = pd.Series(test_y_predicted)

        print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        train_sizes, train_scores, test_scores = learning_curve(
            learner,
            training_x,
            training_y, n_jobs=-1,
            cv=10,
            train_sizes=np.linspace(.1, 1.0, 10),
            random_state=seed)
        v_train_scores, v_test_scores = validation_curve(learner,
                                                     training_x,
                                                     training_y,
                                                     param_name="max_depth",
                                                     param_range=max_depths,
                                                     cv=10,
                                                     n_jobs=-1)

    #learning Curve
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

    plt.show()

    #validation curve
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(v_train_scores, axis=1)
    train_std = np.std(v_train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(v_test_scores, axis=1)
    test_std = np.std(v_test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(max_depths, train_mean, label="Training score", color="black")
    plt.plot(max_depths, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(max_depths, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(max_depths, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve Max Depth vs Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()


    t_d = datetime.now() - t
    timings['DT'] = t_d.seconds


if __name__== "__main__":
  main()
