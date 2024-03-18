import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor
from herder_model import HerderModel


class Model:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.models = {}

        # Include an instance of the HerderModel when initializing
        self.herder_model = HerderModel(self.preprocessor)

    def add_model(self, model_name, features):
        if model_name != "herder":
            self.models[model_name] = {
                "features": features,
                "results": [],
                "estimator": LogisticRegression(solver='liblinear', random_state=42)
            }
        else:
            # Handle Herder model differently, no need to define features or estimator here
            self.models[model_name] = {
                "features": features,
                "results": [],
                "custom_model": self.herder_model  # Direct reference to the HerderModel instance
            }

    def reset_models(self):
        """Clears the models dictionary to remove old models and their results."""
        print("Resetting models...")
        self.models = {}

    def train_models(self):
        X, y = self.preprocessor.load_and_transform_data()
        trained_models = {}

        for model_name, model_info in self.models.items():
            print(f"\nTraining {model_name} model...")
            if model_name != "herder":
                X_selected = X[model_info["features"]]
                estimator = clone(model_info["estimator"])
                self.train_with_cross_validation(X_selected, y, estimator, model_name)
                model_info["estimator"] = estimator
                trained_models[model_name] = model_info["estimator"]
            else:
                # For the Herder model, call its fit_and_evaluate method directly
                self.herder_model.fit(X, y)
                self.herder_model.print_model_formulae()
                trained_models[model_name] = model_info["custom_model"]

        return trained_models

    def calculate_and_store_metrics(self, model_name, fold_metrics, ppv_target=0.98):
        # Initialize lists to collect metrics
        accuracies_before, precisions_before, recalls_before, f1_scores_before = [], [], [], []
        accuracies_after, precisions_after, recalls_after, f1_scores_after = [], [], [], []
        thresholds, train_roc_aucs, val_roc_aucs = [], [], []

        # ROC data initialization
        roc_data = {'train': {'fpr': [], 'tpr': [], 'roc_auc': []},
                    'validation': {'fpr': [], 'tpr': [], 'roc_auc': []}}

        for metrics in fold_metrics:
            # Unpack metrics
            y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba, \
            train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc = metrics

            # Calculate metrics before applying PPV threshold
            accuracies_before.append(accuracy_score(y_test, y_test_pred))
            precisions_before.append(precision_score(y_test, y_test_pred, zero_division=0))
            recalls_before.append(recall_score(y_test, y_test_pred))
            f1_scores_before.append(f1_score(y_test, y_test_pred))

            # Store train and validation ROC AUC for metrics calculation
            train_roc_aucs.append(train_roc_auc)
            val_roc_aucs.append(val_roc_auc)

            # Determine PPV threshold using precision_recall_curve
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
            threshold_indices = np.where(precision >= ppv_target)[0]
            threshold = thresholds_pr[threshold_indices[0] - 1] if threshold_indices.size > 0 else 1.0
            thresholds.append(threshold)

            y_pred_adjusted = (y_test_proba >= threshold).astype(int)

            # Calculate metrics after applying PPV threshold
            accuracies_after.append(accuracy_score(y_test, y_pred_adjusted))
            precisions_after.append(precision_score(y_test, y_pred_adjusted, zero_division=0))
            recalls_after.append(recall_score(y_test, y_pred_adjusted))
            f1_scores_after.append(f1_score(y_test, y_pred_adjusted))

            # Store ROC curve data
            roc_data['train']['fpr'].append(train_fpr)
            roc_data['train']['tpr'].append(train_tpr)
            roc_data['train']['roc_auc'].append(train_roc_auc)
            roc_data['validation']['fpr'].append(val_fpr)
            roc_data['validation']['tpr'].append(val_tpr)
            roc_data['validation']['roc_auc'].append(val_roc_auc)

        # Aggregate and store metrics
        avg_threshold = np.mean(thresholds)
        self.models[model_name]['metrics'] = {
            'before_threshold': {
                "Accuracy": np.mean(accuracies_before),
                "Precision": np.mean(precisions_before),
                "Recall": np.mean(recalls_before),
                "F1": np.mean(f1_scores_before),
                "Train ROC AUC": np.mean(train_roc_aucs),
                "Validation ROC AUC": np.mean(val_roc_aucs),
            },
            'after_threshold': {
                "Accuracy": np.mean(accuracies_after),
                "Precision": np.mean(precisions_after),
                "Recall": np.mean(recalls_after),
                "F1": np.mean(f1_scores_after),
                "Train ROC AUC": np.mean(train_roc_aucs),
                "Validation ROC AUC": np.mean(val_roc_aucs),
            },
            'ppv_threshold': avg_threshold,
        }
        self.models[model_name]['roc_data'] = roc_data

        # Calculate standard deviations for Train ROC AUC and Validation ROC AUC
        std_metrics = {
            "Train ROC AUC STD": np.std(train_roc_aucs),
            "Validation ROC AUC STD": np.std(val_roc_aucs)
        }

        # Store the standard deviation metrics
        self.models[model_name]['metrics_std'] = std_metrics

        # Print summary of metrics including Train and Validation ROC AUC under each threshold section
        print(f"\nMetrics for {model_name} Model (Before PPV Threshold):")
        for metric, value in self.models[model_name]['metrics']['before_threshold'].items():
            print(f"{metric}: {value:.4f}")

        print(f"\nMetrics for {model_name} Model (After PPV Threshold):")
        for metric, value in self.models[model_name]['metrics']['after_threshold'].items():
            print(f"{metric}: {value:.4f}")

        print(f"\nAverage PPV Threshold used for {model_name}: {avg_threshold:.4f}")

        # Print standard deviations for Train and Validation ROC AUC
        print(f"\nStandard Deviations:")
        for metric, value in std_metrics.items():
            print(f"{metric}: {value:.4f}")

    def train_with_cross_validation(self, X, y, estimator, model_name, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_metrics = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_train_proba = estimator.predict_proba(X_train)[:, 1] if hasattr(estimator, "predict_proba") else []

            y_test_pred = estimator.predict(X_test)
            y_test_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else []

            # Calculate and store ROC curve data for both train and validation sets
            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
            train_roc_auc = roc_auc_score(y_train, y_train_proba)

            val_fpr, val_tpr, _ = roc_curve(y_test, y_test_proba)
            val_roc_auc = roc_auc_score(y_test, y_test_proba)

            fold_metrics.append((y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba, train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc))
            self.models[model_name]["results"].append((y_test, y_test_pred, y_test_proba))

        self.calculate_and_store_metrics(model_name, fold_metrics)

        estimator.fit(X, y)
        self.models[model_name]['estimator'] = estimator

    def apply_rfe_feature_selection(self, model_name, step=3, cv=5, scoring='roc_auc'):
        """
        Applies Recursive Feature Elimination (RFE) with cross-validation to select features.

        Parameters:
        - model_name: The name of the model to apply RFE feature selection.
        - step: The number of features to remove at each iteration. Default is 1.
        - cv: The number of folds for cross-validation to select the optimal number of features. Default is 5.
        - scoring: A string to determine the scoring criterion. Default is 'accuracy'.
        """
        print(f"\nApplying RFE Feature Selection for {model_name} model...")

        if model_name not in self.models:
            print(f"Model {model_name} not found. Please add the model first.")
            return

        # Load and transform data
        X, y = self.preprocessor.load_and_transform_data()

        if model_name == 'herder':
            # Perform feature selection specifically for the HerderModel
            self.herder_model.select_features(X, y)
            updated_mcp_features = self.herder_model.mcp_features
            updated_herder_features = self.herder_model.herder_features
            all_selected_features = updated_mcp_features + updated_herder_features
            self.models[model_name]['features'] = all_selected_features
            print(f"Updated features for {model_name}: {all_selected_features}")
        else:
            # For other models, proceed with the usual RFE feature selection process
            X_selected = X[self.models[model_name]["features"]]
            base_model = LogisticRegression(solver='liblinear', random_state=42)
            rfe_cv = RFECV(estimator=base_model, step=step, cv=cv, scoring=scoring)
            rfe_cv.fit(X_selected, y)
            selected_features = X_selected.columns[rfe_cv.support_]
            print(f"Selected features by RFE for {model_name}: {list(selected_features)}")
            self.models[model_name]['features'] = list(selected_features)

    def get_updated_ensemble_features(self):
        # Then, proceed with aggregating ensemble features
        ensemble_features = set()
        for model_name, model_info in self.models.items():
            selected_features = model_info.get('features', [])
            ensemble_features.update(selected_features)
        return list(ensemble_features)

    def plot_roc_curves(self, model_name, data_type='validation'):
        """
        Plot ROC curves for the specified model.

        Parameters:
            model_name (str): The name of the model to plot ROC curves for.
            data_type (str): The type of data to plot ('train', 'validation', or 'both').
        """
        if 'roc_data' not in self.models[model_name]:
            print("ROC curve data not available for this model.")
            return

        roc_data = self.models[model_name]['roc_data']
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))

        if data_type in ['train', 'both']:
            self._plot_roc_curve(ax, roc_data['train'], mean_fpr, 'Training')

        if data_type in ['validation', 'both']:
            self._plot_roc_curve(ax, roc_data['validation'], mean_fpr, 'Validation')

        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'ROC Curve for {model_name}')
        ax.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.show()

    def _plot_roc_curve(self, ax, data, mean_fpr, label_prefix):
        """
        Helper function to plot ROC curve for a given dataset (training or validation).

        Parameters:
            ax (matplotlib.axes._subplots.AxesSubplot): The subplot to plot on.
            data (dict): ROC data for either training or validation set.
            mean_fpr (np.array): Array of mean false positive rates.
            label_prefix (str): Prefix for the curve label to indicate data type.
        """
        tprs = []
        aucs = data['roc_auc']

        for i in range(len(data['fpr'])):
            fpr = data['fpr'][i]
            tpr = data['tpr'][i]
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'{label_prefix} Fold {i+1} (AUC = {aucs[i]:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)

        ax.plot(mean_fpr, mean_tpr, label=f'Mean {label_prefix} ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2, label=f'{label_prefix} ± 1 std. dev.')

    def generate_shap_plot(self, model_name, features):
        model_info = self.models.get(model_name)
        if model_info is None:
            print(f"Model '{model_name}' not found.")
            return

        X, _ = self.preprocessor.load_and_transform_data()
        model = model_info['estimator']
        features = model_info['features']

        # Select only the columns specified as features
        X_selected = X[features]

        explainer = shap.Explainer(model.predict, X_selected)
        shap_values = explainer(X_selected)
        shap.plots.bar(shap_values, max_display = 30)
        shap.plots.beeswarm(shap_values, max_display=30, show=False)

        plt.title(f"{model_name} SHAP Beeswarm Plot", fontsize=20)
        plt.show()

    def plot_prediction_histograms(self, model_name):
        if model_name not in self.models:
            print(f"No model found with the name {model_name}")
            return

        if 'results' not in self.models[model_name] or not self.models[model_name]['results']:
            print(f"No predictions available for the {model_name} model")
            return

        if 'metrics' not in self.models[model_name] or 'ppv_threshold' not in self.models[model_name]['metrics']:
            print(f"PPV threshold not calculated for the {model_name} model")
            return

        plt.figure(figsize=(15, 7))

        true_labels = np.concatenate([result[0] for result in self.models[model_name]["results"]])
        predictions = np.concatenate([result[2] for result in self.models[model_name]["results"]])

        # Retrieve the PPV threshold used for this model
        ppv_threshold = self.models[model_name]['metrics']['ppv_threshold']

        # Plot histograms for true negative and true positive predictions
        sns.histplot(predictions[true_labels == 0], bins=20, stat="density", kde=True, color='blue', alpha=0.5, label='Negative Class')
        sns.histplot(predictions[true_labels == 1], bins=20, stat="density", kde=True, color='red', alpha=0.7, label='Positive Class')

        # Add a vertical line to indicate the PPV threshold
        plt.axvline(x=ppv_threshold, color='green', linestyle='--', label=f'PPV Threshold: {ppv_threshold:.2f}')

        plt.title(f'Prediction Probability Distribution for {model_name} Model', fontsize=20)
        plt.xlabel('Predicted Probability of Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

    def get_logistic_regression_formula(self, model_name):
        # Check if the model exists
        if model_name not in self.models:
            return f"No model found with the name {model_name}"

        model_info = self.models[model_name]
        estimator = model_info['estimator']

        # Check if the estimator is a logistic regression model
        if not hasattr(estimator, 'coef_') or not hasattr(estimator, 'intercept_'):
            return "The specified model does not support this operation. This function is only applicable to logistic regression models."

        # Retrieve the coefficients and intercept
        coefficients = estimator.coef_[0]
        intercept = estimator.intercept_[0]

        # Format the formula
        feature_names = model_info['features']
        terms = [f"{coef:.4f}*{feature}" for coef, feature in zip(coefficients, feature_names)]
        formula = "logit(p) = " + f"{intercept:.4f} + " + " + ".join(terms)

        return formula

