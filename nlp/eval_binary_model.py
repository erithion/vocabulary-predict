import tensorflow as tf
import sys
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#from sklearn import metrics
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report  
from colorama import init, Fore, Back, Style
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from gensim.models.keyedvectors import KeyedVectors

init()

# taken from sklearn example
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
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
    return plt

# target - array of zeros and ones
# threshold - less or greater than threshold will print a warning
def skewedCheck (target, threshold=0.25, do_exit=False):
    i = sum(target) / target.size
    if threshold <= i <= (1 - threshold):
        print(Fore.GREEN + '%2.2f%% of positive targets' % (i*100))
        print(Style.RESET_ALL)
    else:
        print(Fore.RED + '%2.2f%% of positive targets' % (i*100))
        print(Style.RESET_ALL)
        if do_exit:
            sys.exit('You asked not stop if the data is skewed')

cancer = datasets.load_breast_cancer()

#   print the names of the 13 features
# print("Features: ", cancer.feature_names) # 
#   print the label type of cancer('malignant' 'benign')
# print("Labels: ", cancer.target_names)


skewedCheck(cancer.target)

# no shuffling as we expect articles for word2vec
X_train, X_eval, y_train, y_eval = train_test_split(cancer.data, cancer.target, test_size=0.2,random_state=109) # 80% training

pipe = Pipeline(steps=[('scaler', StandardScaler()), ('estimator', SVC())]) # estimator is just to create a key
tuned_param = [{
                'estimator':[SVC()],
                'estimator__C': [0.01, 0.1, 1, 10, 100, 1000],
                'estimator__gamma': [0.001, 0.0001],
                'estimator__kernel': ['rbf', 'linear']
                },
                {
                'estimator':[LogisticRegression()],
                'estimator__C': [0.1, 1, 10, 100, 1000],
                'estimator__solver': ['sag'],
                'estimator__max_iter' : [15000] # seems there's no way to vary the gradient descent step, have to use many iterations to get the loss function gradient converged
                },
              ]

score = 'f1_micro'
print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(pipe, tuned_param, n_jobs=-1, cv=5, scoring=score, iid=False)
clf.fit(X_train, y_train)

print(Fore.YELLOW + "Best score and its hyper-parameters")
print("%0.3f (+/-%0.03f) for %r"
            % ( clf.cv_results_['mean_test_score'][clf.best_index_]
              , clf.cv_results_['std_test_score'][clf.best_index_] * 2
              , clf.cv_results_['params'][clf.best_index_]))
print(Style.RESET_ALL)

if False: # for future 'verbose' cmdline param
    print("Score details (dev-set)")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
    print()
    
print("Evaluaion (test-set)")
y_true, y_pred = y_eval, clf.predict(X_eval)
print(classification_report(y_true, y_pred))
print()

# Learning curve plot
clf2 = clone(clf.best_estimator_.named_steps['estimator']) # the result is yet unfitted estimator with the same initial params
pipe2 = make_pipeline(StandardScaler(), clf2)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(pipe2, "Learning Curve", X_train, y_train, (0.7, 1.01), cv=cv, n_jobs=-1)
plt.show()    
    
print ("In case the scores are bad consult https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html")


word_vectors = KeyedVectors.load_word2vec_format('../../Git/llearn/data/58/model.txt', binary=False)
