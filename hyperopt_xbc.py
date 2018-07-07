import warnings
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

class Hyperopt_xbc:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = None
        self.best_acc = 0
        self.space = {
                'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
                'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
                'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
                'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
                'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
                'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
                'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
                }
        self.max_evals = 50

    def train_test(self, params):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.clf = XGBClassifier(**params)
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
