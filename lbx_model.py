import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     learning_curve)

from data_preprocessing import DataPreprocessor


class LBxModel:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.model_lc = LogisticRegression(solver='liblinear', random_state=42)
        self.model_nsclc = LogisticRegression(solver='liblinear', random_state=42)
        self.lc_results = []
        self.nsclc_results = []
        self.lc_features = [
            'remainder__CA125', 'remainder__CA15.3',
            'remainder__CEA', 'remainder__CYFRA 21-1', 'remainder__HE4',
            'remainder__NSE', 'remainder__NSE corrected for H-index',
            'remainder__proGRP', 'remainder__SCCA'
        ]
        self.nsclc_features = [
            'remainder__CEA', 'remainder__CYFRA 21-1', 'remainder__NSE',
            'remainder__proGRP'
        ]

    def train_models(self):
        X, y = self.preprocessor.load_and_transform_data()
        X_lc = X[self.lc_features]
        X_nsclc = X[self.nsclc_features]

        #Hyperparameter tuning for LC model
        print("Tuning hyperparameters for LC model...")
        best_estimator_lc = self.tune_hyperparameters(X_lc, y)
        self.model_lc = self.train_with_cross_validation(X_lc, y, best_estimator_lc)

        print("Generating SHAP feature importance plot for LC model...")
        self.generate_shap_plot(X_lc, self.model_lc)

        # Hyperparameter tuning for NSCLC model
        print("Tuning hyperparameters for NSCLC model...")
        best_estimator_nsclc = self.tune_hyperparameters(X_nsclc, y)
        self.model_nsclc = self.train_with_cross_validation(X_nsclc, y, best_estimator_nsclc)

        print("Generating SHAP feature importance plot for NSCLC model...")
        self.generate_shap_plot(X_nsclc, self.model_nsclc)

    def generate_shap_plot(self, X, model):
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values)

    def train_with_cross_validation(self, X, y, estimator, n_splits=5, model_type='lc'):
        """
        Trains and evaluates the model using Stratified K-Fold Cross Validation.
        Stores fold results in the appropriate class attribute based on the model type.

        Parameters:
        - X: Features dataframe.
        - y: Target vector.
        - estimator: The machine learning estimator to be trained.
        - n_splits: Number of folds for cross-validation.
        - model_type: Indicates the type of model being trained ('lc' or 'nsclc').
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        best_score = 0
        best_model = None

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = clone(estimator)  # Clone the estimator to ensure it's a fresh model for each fold
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
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

            # Store results for the correct model type
            if model_type == 'lc':
                self.lc_results.append((y_test, proba))
            elif model_type == 'nsclc':
                self.nsclc_results.append((y_test, proba))

        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        print(f"Best ROC AUC Score for {model_type.upper()} Model: {best_score:.4f}")
        for metric, score in avg_scores.items():
            print(f"{metric.capitalize()} (average) for {model_type.upper()}: {score:.4f}")

        return best_model

    def tune_hyperparameters(self, X, y):
        # Define the parameter grid for 'C'
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

        # Initialize the Logistic Regression model
        log_reg = LogisticRegression(solver='liblinear', random_state=42)

        # Setup the grid search with cross-validation
        grid_search = GridSearchCV(log_reg, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='roc_auc', verbose=1)

        # Fit the grid search to the data
        grid_search.fit(X, y)

        # Output the best parameters and the best score
        print("Best parameters:", grid_search.best_params_)
        print("Best ROC AUC score:", grid_search.best_score_)

        return grid_search.best_estimator_

    def plot_learning_curve(self, X, y, title):
        train_sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(solver='liblinear', random_state=42),
            X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))

        # Compute mean and standard deviation for training set scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        # Compute mean and standard deviation for test set scores
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.grid()

        plt.show()


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

    def plot_prediction_histograms(self):
        """
        Plot histograms of the predicted probabilities for the LC and NSCLC models with improved aesthetics.
        """
        # Set the visual theme for the seaborn plots
        sns.set(style="whitegrid")

        # Plot for LC model
        plt.figure(figsize=(15, 7))
        lc_prediction = np.concatenate([proba for _, proba in self.lc_results])
        lc_y_true = np.concatenate([y_test for y_test, _ in self.lc_results])

        sns.histplot(lc_prediction[lc_y_true == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(lc_prediction[lc_y_true == 1], bins=20, kde=True, label='Positives', alpha=0.7, color='red')
        plt.title('Probability Distribution for LC Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()

        # Plot for NSCLC model
        plt.figure(figsize=(15, 7))
        nsclc_prediction = np.concatenate([proba for _, proba in self.nsclc_results])
        nsclc_y_true = np.concatenate([y_test for y_test, _ in self.nsclc_results])

        sns.histplot(nsclc_prediction[nsclc_y_true == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(nsclc_prediction[nsclc_y_true == 1], bins=20, kde=True, label='Positives', alpha=0.7, color='red')
        plt.title('Probability Distribution for NSCLC Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlim(0, 1)
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

        lc_formula = self.get_model_formula(self.model_lc, ['CYFRA 21-1', 'CEA', 'HE4'])
        nsclc_formula = self.get_model_formula(self.model_nsclc, ['CEA', 'CYFRA 21-1', 'NSE', 'proGRP', 'HE4'])

        print("LC Model Formula:\n", lc_formula)
        print("\nNSCLC Model Formula:\n", nsclc_formula)

