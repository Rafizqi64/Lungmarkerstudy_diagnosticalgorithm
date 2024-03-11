import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_preprocessing import DataPreprocessor


class BrockAndHerderModel:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.fold_results = []
        self.model_brock = LogisticRegression(solver='liblinear', random_state=42)
        self.model_herder = LogisticRegression(solver='liblinear', random_state=42)
        self.brock_features = [
            'cat__Nodule Type_GroundGlass',
            'cat__Nodule Type_PartSolid',
            'cat__Nodule Type_Solid',
            'remainder__Family History of LC',
            'remainder__Current/Former smoker',
            'remainder__Emphysema',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Nodule Count',
            'remainder__Spiculation',
        ]

        self.herder_features = [
            'cat__PET-CT Findings_Faint',
            'cat__PET-CT Findings_Intense',
            'cat__PET-CT Findings_Moderate',
            'cat__PET-CT Findings_No FDG avidity',
            'remainder__Family History of LC',
            'remainder__Previous History of Extra-thoracic Cancer',
            'remainder__Emphysema',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Nodule Count',
            'remainder__Spiculation',
        ]

    def train_models(self):
        # Load and preprocess the data
        X, y = self.preprocessor.load_and_transform_data()
        X_brock = X[self.brock_features]
        X_herder = X[self.herder_features]

        # Hyperparameter tuning for brock model
        # print("Tuning hyperparameters for brock model...")
        # best_estimator_brock = self.tune_hyperparameters(X_brock, y)

        self.fold_results = []
        self.model_brock = self.train_with_cross_validation(X_brock, y)
        self.generate_shap_plot(X_brock, self.model_brock)
        brock_results = self.fold_results.copy()  # Make a copy for plotting

        # Hyperparameter tuning for herder model
        # print("Tuning hyperparameters for LC model...")
        # best_estimator_lc = self.tune_hyperparameters(X_herder, y)

        # Reset fold_results and train Herder model
        self.fold_results = []
        self.model_herder = self.train_with_cross_validation(X_herder, y)
        self.generate_shap_plot(X_herder, self.model_herder)
        herder_results = self.fold_results.copy()  # Make a copy for plotting

        # Store or otherwise handle the results for plotting
        self.brock_results = brock_results
        self.herder_results = herder_results


    def generate_shap_plot(self, X, model):
        # Assuming X is your feature matrix with proper column names
        # and 'model' is a fitted scikit-learn model or compatible object.
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)

        # Check the shape of the SHAP values to ensure they match your expectations

        # Plotting the SHAP values
        shap.plots.beeswarm(shap_values)

    def train_with_cross_validation(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        best_score = 0
        best_model = None

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = LogisticRegression(solver='liblinear', random_state=42) # Clone the estimator to ensure a fresh model for each fold
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # Calculate and store scores
            scores['accuracy'].append(accuracy_score(y_test, predictions))
            scores['precision'].append(precision_score(y_test, predictions))
            scores['recall'].append(recall_score(y_test, predictions))
            scores['f1'].append(f1_score(y_test, predictions))
            scores['roc_auc'].append(roc_auc_score(y_test, proba))

            # Update best model if this fold is better
            if roc_auc_score(y_test, proba) > best_score:
                best_score = roc_auc_score(y_test, proba)
                best_model = model

        # After all splits, calculate average scores
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        print(f"Best ROC AUC Score: {best_score}")
        for metric, score in avg_scores.items():
            print(f"{metric.capitalize()} (average): {score:.4f}")

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
        plt.figure(figsize=(15, 7))

        # Brock model
        brock_prediction = np.concatenate([proba for _, proba in self.brock_results])
        brock_y_true = np.concatenate([y_test for y_test, _ in self.brock_results])

        sns.histplot(brock_prediction[brock_y_true == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(brock_prediction[brock_y_true == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        plt.title('Probability Distribution for Brock Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.show()

        plt.figure(figsize=(15, 7))

        # Herder model
        herder_prediction = np.concatenate([proba for _, proba in self.herder_results])
        herder_y_true = np.concatenate([y_test for y_test, _ in self.herder_results])

        sns.histplot(herder_prediction[herder_y_true == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(herder_prediction[herder_y_true == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        plt.title('Probability Distribution for Herder Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.show()

    def get_models(self):
        print('Brock and Herder models finished training')
        return self.model_brock, self.model_herder


