import warnings
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

class Hyperopt_gbc:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = None
        self.best_acc = 0
        self.space = {
              'max_depth': hp.choice('max_depth', range(1,20)),
              'max_features': hp.choice('max_features', range(1,5)),
              'min_samples_leaf': hp.choice('min_samples_leaf', range(1,50)),
              'min_samples_split': hp.choice('min_samples_split', range(10, 50, 10)),
              'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2,50)),
              'loss': hp.choice('loss', ['deviance', 'exponential']),
              'n_estimators': hp.choice('n_estimators', range(1,500)),
              'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
              'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01)
               }
        self.max_evals = 50

    def train_test(self, params):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.clf = GradientBoostingClassifier(**params)
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
