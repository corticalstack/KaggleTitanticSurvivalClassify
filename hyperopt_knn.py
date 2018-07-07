import warnings
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class Hyperopt_knn:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = None
        self.best_acc = 0
        self.space = {
              'n_neighbors': hp.choice('n_neighbors', range(1,100))
                }
        self.max_evals = 50

    def train_test(self, params):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.clf = KNeighborsClassifier(**params)
        self.clf.fit(self.X, self.y)
        return cross_val_score(self.clf, self.X, self.y, cv=10).mean()
    
    def f(self, params):
        acc = self.train_test(params)
        if acc > self.best_acc:
            self.best_acc = acc
        return {'loss': -acc, 'status': STATUS_OK}
    
    def best(self):
        trials = Trials()
        best = fmin(self.f, self.space, algo=tpe.suggest, max_evals = self.max_evals, trials=trials)             
        self.clf.set_params(**best)
        return self.clf, space_eval(self.space, best), self.best_acc
