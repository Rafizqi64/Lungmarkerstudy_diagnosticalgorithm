import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
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

    def print_scores(self):
        best_roc_auc_score = self.cv_scores['roc_auc']
        print(f"Best Average ROC AUC Score: {best_roc_auc_score:.4f}")
        for metric, score in self.cv_scores.items():
            print(f"{metric.capitalize()}: {score:.4f}")


