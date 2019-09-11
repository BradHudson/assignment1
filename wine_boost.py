import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    df = pd.read_csv("Dataset/winedata.csv", delimiter=";")

    indeksDaarlig = df.loc[df['quality'] <= 6].index
    indeksGod = df.loc[df['quality'] > 6].index
    df.iloc[indeksDaarlig, df.columns.get_loc('quality')] = 0
    df.iloc[indeksGod, df.columns.get_loc('quality')] = 1
    seed = 200
    np.random.seed(seed)

    X = np.array(df.iloc[:, 0:-1])
    Y = np.array(df.iloc[:, -1])

    training_x, testing_x, training_y, testing_y = train_test_split(X, Y, test_size=0.4, random_state=seed, shuffle=True)

    learner = DecisionTreeClassifier(max_depth=2,random_state=seed)
    boosted_learner = AdaBoostClassifier(algorithm='SAMME', base_estimator=learner, random_state=seed)
    boosted_learner.fit(training_x, training_y)
    print('Training Score: ' + str(boosted_learner.score(training_x,training_y)))
    print('Testing Score: ' + str(boosted_learner.score(testing_x, testing_y)))
    test_y_predicted = boosted_learner.predict(testing_x)
    y_true = pd.Series(testing_y)
    y_pred = pd.Series(test_y_predicted)

    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    train_sizes, train_scores, test_scores = learning_curve(
        boosted_learner,
        training_x,
        training_y, n_jobs=-1,
        cv=10,
        train_sizes=np.linspace(.1, 1.0, 10),
        random_state=seed)

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

    plt.savefig('wineboostedlearningcurve.png')
    plt.close()

    max_depth_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    for i in range(1, 50):
        max_depth_array.append(i)
        learner2 = DecisionTreeClassifier(max_depth=i + 1, random_state=seed)
        boosted_learner2 = AdaBoostClassifier(algorithm='SAMME', base_estimator=learner2, random_state=seed)
        cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=3).mean())

        boosted_learner2.fit(training_x, training_y)
        training_depth_array.append(boosted_learner2.score(training_x, training_y))
        testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))

    plt.plot(max_depth_array, training_depth_array, label='Training')
    plt.plot(max_depth_array, testing_depth_array, label='Testing')
    plt.plot(max_depth_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Accuracy vs Max Depth")
    plt.ylabel('Accuracy %')
    plt.xlabel('Max Depth')
    plt.xlim([1, 50])
    plt.savefig('wineboostedmaxdepth.png')
    plt.close()

    estimator_array = []
    training_depth_array = []
    testing_depth_array = []
    cross_val_score_array = []

    learner2 = DecisionTreeClassifier(max_depth= 2,random_state=seed)
    for i in range(1, 50):
        estimator_array.append(i)
        boosted_learner2 = AdaBoostClassifier(algorithm='SAMME', base_estimator=learner2, random_state=seed,
                                              n_estimators=i + 1)
        cross_val_score_array.append(cross_val_score(boosted_learner2, training_x, training_y, cv=3).mean())

        boosted_learner2.fit(training_x, training_y)
        training_depth_array.append(boosted_learner2.score(training_x, training_y))
        testing_depth_array.append(boosted_learner2.score(testing_x, testing_y))

    plt.plot(estimator_array, training_depth_array, label='Training')
    plt.plot(estimator_array, testing_depth_array, label='Testing')
    plt.plot(estimator_array, cross_val_score_array, label='Cross Validation')
    plt.legend(loc=4, fontsize=8)
    plt.title("Accuracy vs Estimator Count")
    plt.ylabel('Accuracy %')
    plt.xlabel('Number of Estimators')
    plt.xlim([1, 50])
    plt.savefig('wineboostedestimators.png')
    plt.close()


if __name__== "__main__":
  main()
