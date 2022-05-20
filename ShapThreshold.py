import numpy as np


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def find_threshold_balance(score, labels, target_score, source_shap):
        sorted_idx = np.argsort(score)

        sorted_score = score[sorted_idx]
        sorted_labels = labels[sorted_idx]
        print(sorted_score)

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1
            print(np.abs(fp - fn), min_fp_fn)
            if np.abs(fp - fn) < min_fp_fn:
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_score[i]
                print("Thres: ", thres)
                print(np.mean(target_score >= thres) * 100.0)

        return min_fp_fn, thres

    def _loss(self, fp, fn):
        return np.abs(fp - fn)

    def minimize(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_thres = np.zeros(X.shape[1])
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_thres, method="nelder-mead"
        )

    def SHAP_accuracy(
        source_probs,
        source_labels,
        target_probs,
        source_shap,
        target_shap,
        score_function="MC",
    ):

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

        shap_threshold = self.minimize(
            source_shap, np.array(source_labels == source_preds)
        )
        ATC_acc = np.mean(target_score >= ATC_threshold) * 100.0

        return ATC_acc

    def ATC_accuracy(source_probs, source_labels, target_probs, score_function="MC"):

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

        _, ATC_threshold = find_threshold_balance(
            source_score, np.array(source_labels == source_preds)
        )

        ATC_acc = np.mean(target_score >= ATC_threshold) * 100.0

        return ATC_acc

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
