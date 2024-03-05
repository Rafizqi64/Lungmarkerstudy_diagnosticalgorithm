import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (RocCurveDisplay, accuracy_score, auc, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate


class ensemble_model:
    def __init__(self, models, voting='soft'):
        self.ensemble = VotingClassifier(estimators=models, voting=voting)
        self.cv_scores = {}

    def fit_evaluate(self, X, y, n_splits=5, random_state=42, scoring=None):
        if scoring is None:
            scoring = {'accuracy': 'accuracy',
                       'precision': 'precision',
                       'recall': 'recall',
                       'f1' : 'f1',
                       'roc_auc': 'roc_auc'}

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = cross_validate(self.ensemble, X, y, cv=cv, scoring=scoring, return_train_score=False)

        # Calculate and store average scores directly
        for metric, score_name in scoring.items():
            metric_key = 'test_' + score_name
            self.cv_scores[metric] = np.mean(cv_results[metric_key])

    def plot_roc_curve(self, X, y, n_splits=5, random_state=42):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X, y)):
            self.ensemble.fit(X[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                self.ensemble,
                X[test],
                y[test],
                name=f"ROC fold {fold}",
                alpha=0.6,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.3,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC Curve across CV folds for Ensemble model",
        )
        ax.legend(loc="lower right")
        plt.show()

    def print_scores(self):
        best_roc_auc_score = self.cv_scores['roc_auc']
        print(f"Best Average ROC AUC Score: {best_roc_auc_score:.4f}")
        for metric, score in self.cv_scores.items():
            print(f"{metric.capitalize()}: {score:.4f}")


