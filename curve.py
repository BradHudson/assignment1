def Snippet_139():
    print()
    print(format('How to plot a learning Curve in Python', '*^82'))

    # import warnings
    # warnings.filterwarnings("ignore")

    # load libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]
    df = pd.read_csv("Dataset/abalone.data", names=column_names)

    # df['rings'] = df['rings'].apply(lambda x: 'young' if x <= 7.5 else ('adult' if x <= 13 else 'old'))

    for label in "MFI":
        df[label] = df["sex"] == label
    del df["sex"]

    # # Load data
    # digits = load_digits()

    # Create feature matrix and target vector
    # X, y = digits.data, digits.target
    X = df[["M", "F", "I", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
            "shell weight"]].values
    y = df['rings'].values

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(X, y, train_size=0.8, random_state=40)

    # Plot Learning Curve
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(), train_inputs, train_classes, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.subplots(1, figsize=(10, 10))
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout();
    plt.show()


Snippet_139()