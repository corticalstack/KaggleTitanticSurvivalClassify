import warnings
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class Hyperopt_dtc:
    def __init__(self, X, y, seed):
        self.name = 'Decsion Tree'
        self.name_short = 'DTC'
        self.X = X
        self.y = y
        self.seed = seed
        self.clf = None
        self.best_acc = 0
        self.space = {
               'max_depth': hp.choice('max_depth', range(1, 20)),
               'max_features': hp.choice('max_features', range(1, 5)),
               'criterion': hp.choice('criterion', ["gini", "entropy"])
               }
        self.max_evals = 50

    def train_test(self, params):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.clf = DecisionTreeClassifier(**params)
        self.clf.fit(self.X, self.y)
        return cross_val_score(self.clf, self.X, self.y, scoring='roc_auc', cv=10).mean()
    
    def f(self, params):
        acc = self.train_test(params)
        if acc > self.best_acc:
            self.best_acc = acc
        return {'loss': 1-acc, 'status': STATUS_OK}
    
    def best(self):
        trials = Trials()
        best = fmin(self.f, self.space, algo=tpe.suggest, max_evals = self.max_evals, rstate= np.random.RandomState(self.seed), trials=trials)
        self.clf.set_params(**best)
        return self.clf, self.name, self.name_short, space_eval(self.space, best), self.best_acc
