from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score,accuracy_score
# original version
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y):
        return -accuracy_score(y, X>=coef[0])

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Powell')

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_['x']
        return X>=coef

    def coefficients(self):
        return self.coef_['x']