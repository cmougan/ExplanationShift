from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score, accuracy_score

# original version
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _fp_fn(self, coef, X, y):
        fp = np.sum(X[y < coef[0]] >= coef[0])
        fn = np.sum(X[y >= coef[0]] < coef[0])
        return -np.abs(fp - fn)

    def _accuracy(self, coef, X, y):
        return -accuracy_score(y, X >= coef[0])

    def fit(self, X, y):
        loss_partial = partial(self._accuracy, X=X, y=y)
        initial_coef = np.repeat([0.5], X.shape[1])
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method="Powell")

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_["x"]
        return X >= coef

    def coefficients(self):
        return self.coef_["x"]

    def negative_entropy(self, X):
        return np.sum(np.multiply(X, np.log(X + 1e-20)), axis=1)


class ATC(object):
    def ATC_accuracy(
        self, source_probs, source_labels, target_probs, score_function="MC"
    ):
        """
        Calculate the accuracy of the ATC model.
        # Arguments
            source_probs: numpy array of shape (N, num_classes)
            source_labels: numpy array of shape (N, )
            target_probs: numpy array of shape (M, num_classes)
            score_function: string, either "MC" or "NE"
        # Returns
            ATC_acc: float
        """

        if score_function == "MC":
            source_score = np.max(source_probs, axis=-1)
            target_score = np.max(target_probs, axis=-1)

        elif score_function == "NE":
            source_score = np.sum(
                np.multiply(source_probs, np.log(source_probs + 1e-20)), axis=1
            )
            target_score = np.sum(
                np.multiply(target_probs, np.log(target_probs + 1e-20)), axis=1
            )

        source_preds = np.argmax(source_probs, axis=-1)

        _, ATC_threshold = self.find_threshold_balance(
            source_score, source_labels == source_preds
        )

        ATC_acc = np.mean(target_score >= ATC_threshold) * 100.0

        return ATC_acc

    def find_threshold_balance(self, score, labels):
        sorted_idx = np.argsort(score)

        sorted_score = score[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        min_fp = fp
        min_fn = fn
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp = fp
                min_fn = fn
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_score[i]

        return min_fp_fn, thres

    # def fit(self, source_probs, source_labels, score_function="MC"):
    def fit(self, X, Y, score_function="MC"):
        self.score_function = score_function
        if self.score_function == "MC":
            source_score = np.max(X, axis=-1)
        elif self.score_function == "NE":
            source_score = np.sum(np.multiply(X, np.log(X + 1e-20)), axis=1)

        source_preds = np.argmax(X, axis=-1)

        _, self.ATC_threshold = self.find_threshold_balance(
            source_score, Y == source_preds
        )

    # def predict(self, target_probs):
    def predict(self, X):
        if self.score_function == "MC":
            target_score = np.max(X, axis=-1)
        elif self.score_function == "NE":
            target_score = np.sum(np.multiply(X, np.log(X + 1e-20)), axis=1)
        return np.mean(target_score >= self.ATC_threshold) * 100.0


class SATC(object):
    def ATC_accuracy(
        self, source_probs, source_labels, target_probs, score_function="MC"
    ):
        """
        Calculate the accuracy of the ATC model.
        # Arguments
            source_probs: numpy array of shape (N, num_classes)
            source_labels: numpy array of shape (N, )
            target_probs: numpy array of shape (M, num_classes)
            score_function: string, either "MC" or "NE"
        # Returns
            ATC_acc: float
        """

        if score_function == "MC":
            source_score = np.max(source_probs, axis=-1)
            target_score = np.max(target_probs, axis=-1)

        elif score_function == "NE":
            source_score = np.sum(
                np.multiply(source_probs, np.log(source_probs + 1e-20)), axis=1
            )
            target_score = np.sum(
                np.multiply(target_probs, np.log(target_probs + 1e-20)), axis=1
            )

        source_preds = np.argmax(source_probs, axis=-1)

        _, ATC_threshold = self.find_threshold_balance(
            source_score, source_labels == source_preds
        )

        ATC_acc = np.mean(target_score >= ATC_threshold) * 100.0

        return ATC_acc

    def find_threshold_balance(self, score, labels):
        sorted_idx = np.argsort(score)

        sorted_score = score[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        min_fp = fp
        min_fn = fn
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp = fp
                min_fn = fn
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_score[i]

        return min_fp_fn, thres

    # def fit(self, source_probs, source_labels, score_function="MC"):
    def fit(self, shap, preds_hard):
        opt = OptimizedRounder()
        opt.fit(shap, preds_hard)

        return opt.coef_["x"]

    # def predict(self, target_probs):
    def predict(self, X):
        return np.mean(target_score >= self.ATC_threshold) * 100.0
