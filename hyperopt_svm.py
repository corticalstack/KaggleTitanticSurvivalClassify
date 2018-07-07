import warnings
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class Hyperopt_svm:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = None
        self.best_acc = 0
        self.space = {
               #'C': hp.uniform('C', 0, 20),
               #'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
               #'gamma': hp.uniform('gamma', 0, 20)
               'kernel': hp.choice('kernel', ['poly', 'rbf']),
               'gamma': hp.uniform('gamma', 0, 5)
                }
        self.max_evals = 3

    def train_test(self, params):
        print("svm train test")
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        print("svm set params")
        self.clf = SVC(**params)
        print("fit")
        self.clf.fit(self.X, self.y)
        print("svm cross val")
        return cross_val_score(self.clf, self.X, self.y, cv=10).mean()
    
    def f(self, params):
        print("svm f")
        acc = self.train_test(params)
        print(acc)
        if acc > self.best_acc:
            self.best_acc = acc
        return {'loss': -acc, 'status': STATUS_OK}
    
    def best(self):
        print("svm best")
        trials = Trials()
        best = fmin(self.f, self.space, algo=tpe.suggest, max_evals = self.max_evals, trials=trials)
        self.clf.set_params(**best)
        return self.clf, space_eval(self.space, best), self.best_acc
