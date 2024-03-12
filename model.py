import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     StratifiedKFold, cross_val_predict,
                                     learning_curve)

from data_preprocessing import DataPreprocessor


class Model:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.models = {}

    def add_model(self, model_name, features):
        self.models[model_name] = {
            "features": features,
            "results": [],
            "estimator": LogisticRegression(solver='liblinear', random_state=42)
        }

    def reset_models(self):
        """Clears the models dictionary to remove old models and their results."""
        print("Resetting models...")
        self.models = {}

    def train_models(self):
        X, y = self.preprocessor.load_and_transform_data()
        trained_models = {}

        for model_name, model_info in self.models.items():
            print(f"Training {model_name} model...")
            X_selected = X[model_info["features"]]

            # Clone the original estimator to ensure we're working with a fresh model each time
            estimator = clone(model_info["estimator"])

            # Direct training without hyperparameter tuning, but with cross-validation for evaluation
            self.train_with_cross_validation(X_selected, y, estimator, model_name)
            # After evaluation, fit the model on the entire dataset
            model_info["estimator"] = estimator

            trained_models[model_name] = model_info["estimator"]

        return trained_models

    def calculate_and_store_metrics(self, model_name, y, y_pred, y_proba=None):
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='binary')
        recall = recall_score(y, y_pred, average='binary')
        f1 = f1_score(y, y_pred, average='binary')
        roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None

        # Print metrics
        print(f"Metrics for {model_name} Model:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc if roc_auc is not None else 'N/A'}")

        # Store metrics
        self.models[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

    def train_with_cross_validation(self, X, y, estimator, model_name, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize lists to store metrics and ROC curve data
        all_y_preds = []
        all_y_proba = []
        fold_fpr = []
        fold_tpr = []
        fold_roc_auc = []
        fold_predictions = []  # Add this to store predictions

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            y_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else []

            all_y_preds.append(y_pred)
            all_y_proba.append(y_proba)
            fold_predictions.append((y_test, y_proba))

            # Calculate ROC AUC for the current fold
            if len(y_proba) > 0:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)
                fold_fpr.append(fpr)
                fold_tpr.append(tpr)
                fold_roc_auc.append(roc_auc)

        concatenated_y_preds = np.concatenate(all_y_preds)
        concatenated_y_proba = np.concatenate(all_y_proba) if all_y_proba else None

        self.calculate_and_store_metrics(model_name, y, concatenated_y_preds, concatenated_y_proba)

        # Store the ROC curve data for this model
        self.models[model_name]['roc_curve'] = {'fpr': fold_fpr, 'tpr': fold_tpr, 'roc_auc': fold_roc_auc}
        self.models[model_name]["results"] = fold_predictions

        estimator.fit(X, y)  # Retrain on the whole dataset
        self.models[model_name]['estimator'] = estimator

    def plot_roc_curves(self, model_name):
        if 'roc_curve' not in self.models[model_name]:
            print("ROC curve data not available for this model.")
            return

        roc_data = self.models[model_name]['roc_curve']
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = roc_data['roc_auc']
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot each fold's ROC curve
        for i in range(len(roc_data['fpr'])):
            fpr = roc_data['fpr'][i]
            tpr = roc_data['tpr'][i]
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {aucs[i]:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        # Calculate mean and standard deviation for TPR
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)

        # Plot mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})', lw=2, alpha=0.8)

        # Fill the area between mean ± std
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='\u00B1 1 std. dev.')

        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'ROC Curve for {model_name}')
        ax.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.show()

    def generate_shap_plot(self, model_name, features):
        model_info = self.models.get(model_name)
        if model_info is None:
            print(f"Model '{model_name}' not found.")
            return

        X, _ = self.preprocessor.load_and_transform_data()  # Assuming you only need X for SHAP plot
        model = model_info['estimator']

        # Select only the columns specified as features
        X_selected = X[features]

        explainer = shap.Explainer(model.predict, X_selected)
        shap_values = explainer(X_selected)
        shap.plots.beeswarm(shap_values, show=False)

        plt.title(f"{model_name} SHAP Beeswarm Plot", fontsize=20)
        plt.show()

    # def generate_learning_curve(self, model_name, title="Learning Curve"):
        # model_info = self.models.get(model_name)
        # if model_info is None:
            # print(f"Model '{model_name}' not found.")
            # return

        # X, y = self.preprocessor.load_and_transform_data()
        # estimator = clone(model_info['estimator'])

        # train_sizes, train_scores, test_scores = learning_curve(
            # estimator, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            # n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc'
        # )

        # train_scores_mean = np.mean(train_scores, axis=1)
        # train_scores_std = np.std(train_scores, axis=1)
        # test_scores_mean = np.mean(test_scores, axis=1)
        # test_scores_std = np.std(test_scores, axis=1)

        # plt.figure()
        # plt.title(title)
        # plt.xlabel("Training examples")
        # plt.ylabel("Score")
        # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         # train_scores_mean + train_scores_std, alpha=0.1,
                         # color="r")
        # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         # test_scores_mean + test_scores_std, alpha=0.1, color="g")
        # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 # label="Training score")
        # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 # label="Cross-validation score")

        # plt.legend(loc="best")
        # plt.show()

    def plot_prediction_histograms(self, model_name):
        if model_name not in self.models:
            print(f"No model found with the name {model_name}")
            return

        results = self.models[model_name]["results"]
        if not results:
            print(f"No predictions available for the {model_name} model")
            return

        plt.figure(figsize=(15, 7))

        # Concatenate predictions and true labels
        predictions = np.concatenate([proba for _, proba in results])
        true_labels = np.concatenate([y_test for y_test, _ in results])

        # Plot histograms
        sns.histplot(predictions[true_labels == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(predictions[true_labels == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        plt.title(f'Probability Distribution for {model_name} Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.show()
