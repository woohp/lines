import itertools
from numpy import *
from sklearn import cross_validation
from sklearn import svm

class LClassifier(object):
    """
    Wrapper for a classifier to provide the classify method,
    which is same as the predict method but also returns the probability of the estimate.
    """
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, Y, *args, **kwargs):
        return self.clf.fit(X, Y, *args, **kwargs)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X) 
        else:
            return ones(X.shape)

    def score(self, X, Y):
        return self.clf.score(X, Y)

    def classify(self, X):
        Y_predict = self.predict(X)
        prob_predict = self.predict_proba(X)
        return [(y, probs[y]) for y, probs in zip(Y_predict, prob_predict)]


def pick_best_svc_classifier(kernels, Cs, gammas, X, Y, Xnocv=None, Ynocv=None, k=2, class_weight=None):
    """
    Picks the svm.SVC classifier with the best parameters for the given training data.
    """
    classifiers = []
    if 'linear' in kernels:
        classifiers += [svm.SVC(kernel='linear', C=C, probability=True) 
                for C in Cs]
    if 'rbf' in kernels:
        classifiers += [svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True) 
                for C, gamma in itertools.product(Cs, gammas)]
    return pick_best_classifier(classifiers, X, Y, Xnocv, Ynocv, k, class_weight)


def pick_best_classifier(classifiers, X, Y, Xnocv=None, Ynocv=None, k=2, class_weight=None):
    '''
    Use StratifiedKFold to pick that best classifier on the given data.
    An LClassifier wrapper for the best given classifier will be returned.
    '''
    cv = cross_validation.StratifiedKFold(Y, k)
    best_clf = None
    best_num_wrong = len(X)
    best_max_wrong_proba = 1.0

    X = atleast_2d(X)
    Y = atleast_1d(Y)
    if Xnocv is not None:
        Xnocv = atleast_2d(Xnocv)
        Ynocv = atleast_1d(Ynocv)

    for clf in classifiers:
        num_wrong = 0
        max_wrong_proba = 0.0

        if type(clf) != LClassifier:
            clf = LClassifier(clf)

        for includeset, holdoutset in cv:
            if Xnocv is not None:
                X_include = concatenate((X[includeset], Xnocv))
                Y_include = concatenate((Y[includeset], Ynocv))
            else:
                X_include = X[includeset]
                Y_include = Y[includeset]

            X_holdout = X[holdoutset]
            Y_holdout = Y[holdoutset]

            clf.fit(X_include, Y_include, class_weight=class_weight)

            # predict on the holdout set
            Y_predict = clf.predict(X_holdout)
            probas = clf.predict_proba(X_holdout)
            probas_picked = array([probas[y] for y in Y_predict])

            Y_wrong = Y_predict != Y_holdout
            if any(Y_wrong):
                wrong_probas = probas_picked[Y_wrong]
                max_wrong_proba = max(wrong_probas.max(), max_wrong_proba)

            num_wrong += sum(Y_wrong)

        score = 1.0 - num_wrong / len(X)

        if num_wrong < best_num_wrong or (num_wrong == best_num_wrong and max_wrong_proba < best_max_wrong_proba):
            best_clf = clf
            best_num_wrong = num_wrong
            best_max_wrong_proba = max_wrong_proba


    # train on the whole dataset
    if Xnocv is not None:
        X_all = concatenate((X, Xnocv))
        Y_all = concatenate((Y, Ynocv))
    else:
        X_all = X
        Y_all = Y
    best_clf.fit(X_all, Y_all, class_weight=class_weight)

    return best_clf

