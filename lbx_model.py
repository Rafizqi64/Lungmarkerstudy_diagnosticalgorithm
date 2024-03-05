import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor


class LBxModel:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.model_lc = LogisticRegression(solver='liblinear', random_state=42)
        self.model_nsclc = LogisticRegression(solver='liblinear', random_state=42)

    def train_with_cross_validation(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        model = None
        best_score = 0  # Adjust based on which metric you prioritize

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = LogisticRegression(solver='liblinear', random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # Store scores for this fold
            scores['accuracy'].append(accuracy_score(y_test, predictions))
            scores['precision'].append(precision_score(y_test, predictions))
            scores['recall'].append(recall_score(y_test, predictions))
            scores['f1'].append(f1_score(y_test, predictions))
            scores['roc_auc'].append(roc_auc_score(y_test, proba))

        # Calculate average scores across folds
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}

        # Assuming best model selection based on average ROC AUC; adjust as needed
        if avg_scores['roc_auc'] > best_score:
            best_score = avg_scores['roc_auc']
            best_model = model

        print(f"Best Average ROC AUC Score: {best_score}")
        # Optionally print other scores
        for metric, score in avg_scores.items():
            print(f"{metric.capitalize()} (average): {score:.4f}")

        return best_model

    def train_models(self):
        # Load and preprocess the data
        X, y = self.preprocessor.load_and_transform_data()

        lc_features = ['CYFRA 21-1', 'CEA']
        nsclc_features = ['CEA', 'CYFRA 21-1', 'NSE', 'proGRP']
        lc_indices = self.preprocessor.get_feature_indices(lc_features)
        nsclc_indices = self.preprocessor.get_feature_indices(nsclc_features)

        X_lc = X[:, lc_indices]
        X_nsclc = X[:, nsclc_indices]

        self.model_lc = self.train_with_cross_validation(X_lc, y)
        self.model_nsclc = self.train_with_cross_validation(X_nsclc, y)

    def get_models(self):
        print('LBx models finished training')
        return self.model_lc, self.model_nsclc

    #===============#
    # MODEL FORMULA #
    #===============#

    def get_model_formula(self, model, feature_names):
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]
        formula = f"log(p/(1-p)) = {intercept:.4f} "
        formula += " ".join([f"+ {coef:.4f}*{name}" for coef, name in zip(coefficients, feature_names)])
        return formula

    def print_model_formulas(self):
        # Ensure models are trained
        if self.model_lc is None or self.model_nsclc is None:
            print("Models are not trained yet.")
            return

        lc_formula = self.get_model_formula(self.model_lc, ['CYFRA 21-1', 'CEA'])
        nsclc_formula = self.get_model_formula(self.model_nsclc, ['CEA', 'CYFRA 21-1', 'NSE', 'proGRP'])

        print("LC Model Formula:\n", lc_formula)
        print("\nNSCLC Model Formula:\n", nsclc_formula)

