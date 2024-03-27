import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor
from herder_model import HerderModel


class Model:
    def __init__(self, filepath, target, binary_map, threshold_metric="npv"):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.models = {}
        self.n_estimators = 10
        self.herder_model = HerderModel(self.preprocessor)
        self.threshold_metric = threshold_metric
        self.mcp_coefficients = {
            'remainder__Current/Former smoker': 0.7917,
            'remainder__Previous History of Extra-thoracic Cancer': 1.3388,
            'remainder__Nodule size (1-30 mm)': 0.1274,
            'remainder__Nodule Upper Lobe': 0.7838,
            'remainder__Spiculation': 1.0407,
        }
        self.mcp_intercept = -6.8272


    def add_model(self, model_name, features, use_mcp_scores=False):
        if model_name != "herder":
            self.models[model_name] = {
                "features": features,
                "results": {},
                "estimator": LogisticRegression(solver='liblinear', random_state=42),
                "use_mcp_scores": use_mcp_scores,
            }
        else:
            # Handle Herder model differently
            self.models[model_name] = {
                "features": features,
                "results": {},
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
            X_mod = X.copy()
            if model_info["use_mcp_scores"]:
                # Calculate MCP_score and add it to the DataFrame before training
                X_mod['MCP_score'] = self.calculate_mcp_scores(X_mod)

            X_selected = X_mod[model_info["features"]]

            # If MCP_score is used, ensure it's included in the features for training
            if model_info["use_mcp_scores"] and 'MCP_score' not in model_info["features"]:
                X_selected = X_mod[model_info["features"] + ['MCP_score']]

            if model_name != "herder":
                estimator = clone(model_info["estimator"])
                self.train_with_cross_validation(X_selected, y, estimator, model_name)
                model_info["estimator"] = estimator
                trained_models[model_name] = estimator
            else:
                # For the Herder model, call its fit_and_evaluate method directly
                self.herder_model.fit(X, y)
                self.herder_model.print_model_formulae()
                trained_models[model_name] = model_info["custom_model"]

        return trained_models

    def train_with_cross_validation(self, X, y, estimator, model_name, n_splits=5, desired_percentage=0.95):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []  # List to hold metrics for each fold
        probabilities = []
        y_tests = []
        custom_thresholds = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            estimator.fit(X_train, y_train)
            y_train_proba = estimator.predict_proba(X_train)[:, 1]
            y_test_proba = estimator.predict_proba(X_test)[:, 1]
            probabilities.extend(y_test_proba.tolist())
            y_tests.extend(y_test.tolist())

            # Metrics before applying custom threshold
            y_test_pred = (y_test_proba >= 0.5).astype(int)
            metrics_before = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, zero_division=0),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
            }

            # Determine and apply custom threshold
            custom_threshold = self.determine_custom_threshold(y_test, y_test_proba, metric=self.threshold_metric, desired_percentage=desired_percentage)
            custom_thresholds.append(custom_threshold)
            y_test_pred_custom = (y_test_proba >= custom_threshold).astype(int)

            # Metrics after applying custom threshold
            metrics_after = {
                'accuracy': accuracy_score(y_test, y_test_pred_custom),
                'precision': precision_score(y_test, y_test_pred_custom, zero_division=0),
                'recall': recall_score(y_test, y_test_pred_custom),
                'f1_score': f1_score(y_test, y_test_pred_custom),
            }

            # Calculate standard metrics and ROC curve data
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
            train_roc_auc = roc_auc_score(y_train, y_train_proba)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

            # Append collected metrics for this fold
            fold_metrics.append({
                'metrics_before': metrics_before,
                'metrics_after': metrics_after,
                'train_roc_auc': train_roc_auc,
                'test_roc_auc': test_roc_auc,
                'train_fpr': train_fpr,
                'train_tpr': train_tpr,
                'test_fpr': test_fpr,
                'test_tpr': test_tpr,
                'thresholds': thresholds
            })
        if 'results' not in self.models[model_name]:
            self.models[model_name]['results'] = {}
        self.models[model_name]['results']['probabilities'] = probabilities
        self.models[model_name]['results']['y_test'] = y_tests
        self.models[model_name]['results']['thresholds'] = custom_thresholds
        # After collecting metrics for all folds, calculate aggregated metrics
        self.calculate_and_store_metrics(model_name, fold_metrics)


    def print_aggregated_metrics(self, context, metrics):
        print(f"\nAggregated Metrics {context}:")
        for metric, values in metrics.items():
            print(f"{metric.capitalize()}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    def determine_custom_threshold(self, y_test, y_proba, metric='ppv', desired_percentage=0.95):
        if metric == 'ppv':
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            precision = precision[::-1]
            thresholds = thresholds[::-1]
            eligible_indices = np.where(precision >= desired_percentage)[0]
            threshold_index = eligible_indices[-1] if eligible_indices.size > 0 else 0
            ppv_threshold = thresholds[threshold_index]
            return ppv_threshold

        elif self.threshold_metric == 'npv':
            # Convert y_test to a numpy array if it's a pandas Series
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

            # Sort the probabilities and corresponding true labels
            sorted_indices = np.argsort(y_proba)
            sorted_proba = y_proba[sorted_indices]
            sorted_y_test_np = y_test_np[sorted_indices]

            # Initialize variables to track the best threshold and its NPV
            best_threshold = None
            best_npv = 0

            # Iterate over probabilities as potential thresholds
            for idx, threshold in enumerate(sorted_proba[:-1]):  # Exclude the last one to prevent division by zero
                # Predictions based on current threshold
                y_pred = (sorted_proba > threshold).astype(int)

                # Confusion matrix elements
                tn, fp, fn, tp = confusion_matrix(sorted_y_test_np, y_pred).ravel()

                # Calculate NPV
                if (tn + fn) > 0:
                    npv = tn / (tn + fn)
                    if npv >= desired_percentage and npv > best_npv:
                        best_npv = npv
                        best_threshold = threshold
            # Return the best threshold found; default to 0.5 if none found
            return best_threshold if best_threshold is not None else 0.5
        else:
            raise ValueError("Invalid metric specified. Choose 'ppv' or 'npv'.")

    def calculate_mcp_scores(self, X):
        x_prime = self.mcp_intercept
        for feature, coeff in self.mcp_coefficients.items():
            if feature in X.columns:
                x_prime += X[feature] * coeff
            else:
                print(f"Warning: Column {feature} not found in DataFrame.")
        mcp_scores = 1 / (1 + np.exp(-x_prime))
        print(mcp_scores.max())
        return mcp_scores

    def calculate_and_store_metrics(self, model_name, fold_metrics):
        # Initialize containers for aggregated metrics
        aggregated_metrics_before = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        aggregated_metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        aggregated_roc_data = {
            'train_fpr': [], 'train_tpr': [], 'train_roc_auc': [],
            'test_fpr': [], 'test_tpr': [], 'test_roc_auc': []
        }

        # Loop through each fold to aggregate metrics
        for fm in fold_metrics:
            for metric in aggregated_metrics_before:
                aggregated_metrics_before[metric].append(fm['metrics_before'][metric])
                aggregated_metrics_after[metric].append(fm['metrics_after'][metric])
            aggregated_roc_data['train_fpr'].append(fm['train_fpr'])
            aggregated_roc_data['train_tpr'].append(fm['train_tpr'])
            aggregated_roc_data['train_roc_auc'].append(fm['train_roc_auc'])
            aggregated_roc_data['test_fpr'].append(fm['test_fpr'])
            aggregated_roc_data['test_tpr'].append(fm['test_tpr'])
            aggregated_roc_data['test_roc_auc'].append(fm['test_roc_auc'])

        # Calculate averages and standard deviations for metrics
        final_metrics_before = {metric: {'avg': np.mean(values), 'std': np.std(values)} for metric, values in aggregated_metrics_before.items()}
        final_metrics_after = {metric: {'avg': np.mean(values), 'std': np.std(values)} for metric, values in aggregated_metrics_after.items()}
        avg_train_roc_auc = np.mean(aggregated_roc_data['train_roc_auc'])
        avg_test_roc_auc = np.mean(aggregated_roc_data['test_roc_auc'])

        # Store aggregated metrics in the model's dictionary
        self.models[model_name]['metrics_before'] = final_metrics_before
        self.models[model_name]['metrics_after'] = final_metrics_after
        self.models[model_name]['roc_data'] = aggregated_roc_data

        # Optionally, print summary of metrics before and after threshold adjustment
        print(f"Metrics for {model_name} Model (Before Threshold Adjustment):")
        for metric, stats in final_metrics_before.items():
            print(f"{metric.capitalize()}: Avg = {stats['avg']:.4f}, Std = {stats['std']:.4f}")
        print(f"Train ROC AUC: Avg = {avg_train_roc_auc:.4f}")
        print(f"Test ROC AUC: Avg = {avg_test_roc_auc:.4f}\n")
        print(f"\nMetrics for {model_name} Model (After Threshold Adjustment):")
        for metric, stats in final_metrics_after.items():
            print(f"{metric.capitalize()}: Avg = {stats['avg']:.4f}, Std = {stats['std']:.4f}")
        print(f"Train ROC AUC: Avg = {avg_train_roc_auc:.4f}")
        print(f"Test ROC AUC: Avg = {avg_test_roc_auc:.4f}\n")

    def apply_tree_based_feature_selection(self, model_name):
        """
        Applies tree-based feature selection using a Random Forest classifier.
        Directly handles 'herder' model as well.

        Parameters:
        - model_name: The name of the model to apply feature selection.
        """
        print(f"\nApplying Tree-based Feature Selection for {model_name} model...")

        if model_name not in self.models:
            print(f"Model {model_name} not found. Please add the model first.")
            return

        # Load and transform data
        X, y = self.preprocessor.load_and_transform_data()

        if model_name == 'herder':
            X_selected = X[self.herder_model.mcp_features]
        else:
            X_selected = X[self.models[model_name]["features"]]

        # Initialize Random Forest to estimate feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_selected, y)

        # Select features based on importance weights
        selector = SelectFromModel(rf, prefit=True, threshold='mean')
        selected_features_mask = selector.get_support()
        selected_features = X_selected.columns[selected_features_mask]

        print(f"Selected features by Tree-based method for {model_name}: {list(selected_features)}")

        # Update features based on selection
        if model_name == 'herder':
            self.mcp_features = list(selected_features)
        else:
            self.models[model_name]['features'] = list(selected_features)

    def apply_logistic_l1_feature_selection(self, model_name, Cs = np.logspace(-7, 0, 100), cv=5):
        """
        Applies Logistic Regression with L1 regularization for feature selection,
        including custom handling for the 'herder' model.

        Parameters:
        - model_name: The name of the model to apply feature selection.
        - cv: The number of folds for cross-validation to select the optimal C (inverse of regularization strength). Default is 5.
        """
        print(f"\nApplying Logistic Regression with L1 regularization for {model_name} model...")

        if model_name not in self.models:
            print(f"Model {model_name} not found. Please add the model first.")
            return

        # Load and transform data
        X, y = self.preprocessor.load_and_transform_data()

        X_selected = X[self.models[model_name]["features"]]

        # Initialize Logistic Regression with L1 penalty using cross-validation
        logistic_l1_cv = LogisticRegressionCV(cv=cv, penalty='l1', Cs=Cs, solver='liblinear', random_state=42, max_iter=10000)
        logistic_l1_cv.fit(X_selected, y)

        # Identify non-zero coefficients (selected features)
        selected_features_mask = np.sum(np.abs(logistic_l1_cv.coef_), axis=0) != 0
        selected_features = X_selected.columns[selected_features_mask]

        print(f"Selected features by Logistic Regression with L1 for {model_name}: {list(selected_features)}")

        # Update features based on selection
        self.models[model_name]['features'] = list(selected_features)


    def apply_rfe_feature_selection(self, model_name, step=2, cv=5, scoring='roc_auc'):
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
            self.herder_model.select_features_l1(X, y)
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
        # Initialize a set to hold all unique ensemble features
        ensemble_features = set()

        # Iterate through each model and its information
        for model_name, model_info in self.models.items():
            # Add regularly selected features
            selected_features = model_info.get('features', [])
            ensemble_features.update(selected_features)

            # If the model uses MCP scores, ensure MCP-related features are included
            if model_info.get("use_mcp_scores", False):
                mcp_related_features = self.mcp_coefficients.keys()  # Assuming MCP feature names are the keys
                ensemble_features.update(mcp_related_features)

        return list(ensemble_features)

    def plot_prediction_histograms(self, model_name):
        if model_name not in self.models or 'results' not in self.models[model_name]:
            print(f"No predictions or model named '{model_name}' available.")
            return

        results = self.models[model_name]['results']
        if 'y_test' not in results or 'probabilities' not in results:
            print(f"Missing 'y_test' or 'probabilities' in results for model '{model_name}'.")
            return

        true_labels = np.array(results['y_test'])
        probabilities = np.array(results['probabilities'])

        # Calculate the mean custom threshold
        mean_threshold = np.mean(results['thresholds']) if 'thresholds' in results else 0.5

        plt.figure(figsize=(15, 7))
        sns.histplot(probabilities[true_labels == 0], bins=20, kde=True, color='blue', alpha=0.5, label='Negative Class')
        sns.histplot(probabilities[true_labels == 1], bins=20, kde=True, color='red', alpha=0.7, label='Positive Class')

        plt.axvline(x=mean_threshold, color='green', linestyle='--', label=f'Mean Threshold: {mean_threshold:.2f}')

        plt.title(f'Prediction Probability Distribution for {model_name} Model', fontsize=10)
        plt.xlabel('Predicted Probability of Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

    def plot_roc_curves(self, model_name, curve_type='test'):
        if 'roc_data' not in self.models[model_name]:
            print(f"ROC curve data not available for {model_name}.")
            return

        roc_data = self.models[model_name]['roc_data']
        mean_fpr = np.linspace(0, 1, 100)

        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        tprs = []
        mean_auc = 0
        std_auc = 0

        # Select curve type: 'train' or 'test'
        curve_key = curve_type + '_fpr'  # Adjust based on how you stored it

        # Check if the desired data is available
        if curve_key in roc_data:
            for i in range(len(roc_data[curve_key])):
                fpr = roc_data[curve_type + '_fpr'][i]
                tpr = roc_data[curve_type + '_tpr'][i]
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} AUC = {auc_score:.2f}')

                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

                mean_auc += auc_score

            mean_auc /= len(roc_data[curve_key])
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            std_auc = np.std([auc(roc_data[curve_type + '_fpr'][i], roc_data[curve_type + '_tpr'][i]) for i in range(len(roc_data[curve_key]))])

            ax.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean {curve_type.capitalize()} ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
        else:
            print(f"No {curve_type} ROC curve data available for {model_name}.")

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f"ROC Curve for {model_name} ({curve_type.capitalize()})")
        ax.legend(loc="lower right")
        plt.show()


    def generate_shap_plot(self, model_name):
        model_info = self.models.get(model_name)
        if model_info is None:
            print(f"Model '{model_name}' not found.")
            return

        X, _ = self.preprocessor.load_and_transform_data()

        if model_info.get("use_mcp_scores", False):
            # Consistently include MCP_score if required by the model
            X['MCP_score'] = self.calculate_mcp_scores(X)

        # Retrieve the model and ensure its features match the training setup
        model = model_info['estimator']
        features = model_info.get('features', [])

        # Ensure MCP_score is considered if it was part of the training
        if model_info.get("use_mcp_scores", False) and 'MCP_score' not in features:
            features.append('MCP_score')

        # Ensure that the dataset for SHAP includes all necessary features
        X_selected = X[features]

        # Initialize the SHAP explainer with the correctly prepared dataset
        explainer = shap.Explainer(model.predict, X_selected)

        # Generate and plot SHAP values
        shap_values = explainer(X_selected)
        shap.plots.bar(shap_values, max_display=30)
        shap.plots.beeswarm(shap_values, max_display=30, show=False)
        plt.title(f"{model_name} SHAP Beeswarm Plot", fontsize=20)
        plt.show()

    def get_logistic_regression_formula(self, model_name):
        # Check if the model exists
        if model_name not in self.models:
            return "No model found with the name {model_name}."

        model_info = self.models[model_name]
        estimator = model_info['estimator']

        # Check if the estimator is a logistic regression model
        if not hasattr(estimator, 'coef_') or not hasattr(estimator, 'intercept_'):
            return "The specified model does not support this operation. This function is only applicable to logistic regression models."

        # Retrieve the coefficients and intercept
        coefficients = estimator.coef_[0]
        intercept = estimator.intercept_[0]

        feature_names = model_info.get('features', [])

        # Check if MCP scores were used and adjust feature names list accordingly
        if model_info.get("use_mcp_scores", False):
            feature_names.append('MCP_score')

        # Construct terms for the formula
        terms = [f"{coef:.4f}*{feature}" for coef, feature in zip(coefficients, feature_names)]

        formula = "logit(p) = " + f"{intercept:.4f} + " + " + ".join(terms)
        return formula

    def plot_confusion_matrices(self, model_name):
        if model_name not in self.models or 'results' not in self.models[model_name]:
            print(f"No results or necessary data found for the model '{model_name}'.")
            return

        results = self.models[model_name]['results']
        if 'y_test' not in results or 'probabilities' not in results:
            print(f"Missing 'y_test' or 'probabilities' in results for model '{model_name}'.")
            return

        # Retrieve true labels and predicted probabilities
        true_labels = np.array(results['y_test'])
        probabilities = np.array(results['probabilities'])
        average_threshold = np.mean(results['thresholds']) if 'thresholds' in results else 0.5

        # Convert probabilities to binary predictions using the default and the custom average thresholds
        default_predictions = (probabilities >= 0.5).astype(int)
        custom_predictions = (probabilities >= average_threshold).astype(int)

        # Compute confusion matrices for both default and custom threshold predictions
        cm_default = confusion_matrix(true_labels, default_predictions)
        cm_custom = confusion_matrix(true_labels, custom_predictions)

        # Calculate sensitivity and specificity from confusion matrices
        # For the default threshold
        tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
        sensitivity_default = tp_default / (tp_default + fn_default)
        specificity_default = tn_default / (tn_default + fp_default)

        # For the custom threshold
        tn_custom, fp_custom, fn_custom, tp_custom = cm_custom.ravel()
        sensitivity_custom = tp_custom / (tp_custom + fn_custom)
        specificity_custom = tn_custom / (tn_custom + fp_custom)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Plotting the confusion matrix for the default threshold
        sns.heatmap(cm_default, annot=True, fmt="d", cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_xlabel('Predicted labels')
        axes[0].set_ylabel('True labels')
        axes[0].set_title(f'Confusion Matrix {model_name} (Default Threshold 0.5)\nSensitivity: {sensitivity_default:.2f}, Specificity: {specificity_default:.2f}')
        axes[0].set_xticklabels(['Negative', 'Positive'])
        axes[0].set_yticklabels(['Negative', 'Positive'], rotation=0)

        # Plotting the confusion matrix for the custom average threshold
        sns.heatmap(cm_custom, annot=True, fmt="d", cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted labels')
        axes[1].set_title(f'{self.threshold_metric} Threshold {average_threshold:.2f})\nSensitivity: {sensitivity_custom:.2f}, Specificity: {specificity_custom:.2f}')
        axes[1].set_xticklabels(['Negative', 'Positive'])
        axes[1].set_yticklabels(['Negative', 'Positive'], rotation=0)

        plt.tight_layout()
        plt.show()

