import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor


class LBxModel:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.model_lc = LogisticRegression(solver='liblinear', random_state=42)
        self.model_nsclc = LogisticRegression(solver='liblinear', random_state=42)


    def train_models(self):
        # Load and preprocess the data
        X, y = self.preprocessor.load_and_transform_data()

        lc_features = ['CYFRA 21-1', 'CEA']
        nsclc_features = ['CEA', 'CYFRA 21-1', 'NSE', 'proGRP']
        lc_indices = self.preprocessor.get_feature_indices(lc_features)
        nsclc_indices = self.preprocessor.get_feature_indices(nsclc_features)

        X_lc = X[:, lc_indices]
        X_nsclc = X[:, nsclc_indices]


        self.fold_results = []
        self.model_lc = self.train_with_cross_validation(X_lc, y)
        lc_results = self.fold_results.copy()  # Make a copy of the results for later plotting

        # Reset fold_results and train Herder model
        self.fold_results = []
        self.model_nsclc = self.train_with_cross_validation(X_nsclc, y)
        nsclc_results = self.fold_results.copy()  # Make a copy for plotting

        # Store or otherwise handle the results for plotting
        self.lc_results = lc_results
        self.nsclc_results = nsclc_results


    def train_with_cross_validation(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        best_score = 0
        best_model = None
        best_y_test = None
        best_proba = None

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = LogisticRegression(solver='liblinear', random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            self.fold_results.append((y_test, proba))

            # Calculate and store scores
            roc_auc = roc_auc_score(y_test, proba)
            scores['accuracy'].append(accuracy_score(y_test, predictions))
            scores['precision'].append(precision_score(y_test, predictions))
            scores['recall'].append(recall_score(y_test, predictions))
            scores['f1'].append(f1_score(y_test, predictions))
            scores['roc_auc'].append(roc_auc)

            # Update best model if this fold is better
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_y_test = y_test
                best_proba = proba

        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        print(f"Best Average ROC AUC Score: {best_score}")
        for metric, score in avg_scores.items():
            print(f"{metric.capitalize()} (average): {score:.4f}")

        return best_model


    def plot_roc_curves(self, model_results, model_name):
        """
        Plot the ROC curve based on provided model results.
        Parameters:
        model_results (list of tuples): Each tuple contains (y_test, proba) for a fold.
        """
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (y_test, proba) in enumerate(model_results):
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.8, label=f'Fold {fold} ROC (AUC = {roc_auc:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

        tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
        tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='± 1 std. dev.')

        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f"ROC Curve across CV folds for {model_name} model")
        ax.legend(loc='lower right')
        plt.show()

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

