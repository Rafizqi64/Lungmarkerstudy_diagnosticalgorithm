import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from scipy import stats
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
    """
    A class used to manage and operate on various predictive models to handle tasks such as training, evaluation,
    and feature selection for lung cancer prediction models. This class is used to retrain the individual brock herder/herbert and Lbx models
    and parses them to the ensemble model class. This model can also be used to train all logistic regression models.

    Attributes:
    - filepath (str): The path to the dataset file.
    - target (str): The Diagnosis variable.
    - binary_map (dict): A dictionary mapping the diagnosis variable's classes to binary labels.
    - threshold_metric (str, optional): The metric ('ppv' or 'npv') based on which the custom threshold is set. Defaults to 'ppv'.
    - guideline_data (DataFrame, optional): Data containing guideline metrics used for comparative analysis.

    Methods:
    - add_model: Adds a model configuration to the model dictionary.
    - reset_models: Resets the model configurations, removing all stored models and their data.
    - train_models: Trains all configured models using specified methods such as cross-validation.
    - train_with_200x_cross_validation: Uses extended cross-validation to train models for robust metric estimation.
    - train_with_smote_cross_validation: Employs SMOTE for handling class imbalance during model training.
    - train_with_cross_validation: Conducts straightforward cross-validation without synthetic sample generation.
    - determine_custom_threshold: Calculates a decision threshold based on the desired performance metric.
    - calculate_mcp_scores: Computes Model Confidence Prediction scores using logistic regression coefficients.
    - calculate_and_store_metrics: Aggregates performance metrics across different folds and stores them.
    - perform_wilcoxon_signed_rank_test: Applies the Wilcoxon signed-rank test to compare model and guideline metrics.
    - perform_mann_whitney_test: Uses the Mann-Whitney U test for statistical comparison between model outputs and guidelines.
    - apply_tree_based_feature_selection: Selects features based on their importance using a Random Forest classifier.
    - apply_logistic_l1_feature_selection: Utilizes L1 regularization to identify relevant features.
    - apply_rfe_feature_selection: Employs Recursive Feature Elimination to reduce the number of features.
    - get_updated_ensemble_features: Gathers all unique features used across the configured models.
    - plot_prediction_histograms: Visualizes the distribution of predicted probabilities.
    - plot_roc_curves: Displays the Receiver Operating Characteristic curves for model evaluation.
    - generate_shap_plot: Generates SHAP plots to explain the impact of features on model predictions.
    - get_logistic_regression_formula: Constructs a readable formula for logistic regression models.
    - plot_confusion_matrices: Plots confusion matrices to evaluate model performance visually.
    """
    def __init__(self, filepath, target, binary_map, threshold_metric="ppv", guideline_data=None):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.models = {}
        self.n_estimators = 10
        self.herder_model = HerderModel(self.preprocessor)
        self.threshold_metric = threshold_metric
        self.guideline_data = guideline_data
        self.mcp_coefficients = {
            'remainder__Current/Former smoker': 0.7917,
            'remainder__Previous History of Extra-thoracic Cancer': 1.3388,
            'remainder__Nodule size (1-30 mm)': 0.1274,
            'remainder__Nodule Upper Lobe': 0.7838,
            'remainder__Spiculation': 1.0407,
        }
        self.mcp_intercept = -6.8272


    def add_model(self, model_name, features, use_mcp_scores=False, train_method=None):
        """
        Adds a new model configuration to the model dictionary to be used in a voting classifier. This configuration includes
        the features to use, whether to utilize Model Confidence Prediction (MCP) scores, and the training
        method to employ.

        The function differentiates between regular models and the 'herder' model, which is handled specially
        due to its unique requirements and setup.

        Parameters:
        - model_name (str): The name of the model to add. Special behavior if 'herder' is used.
        - features (list of str): List of feature names to be used for this model.
        - use_mcp_scores (bool): Flag to indicate whether MCP scores should be incorporated into the model's features.
        - train_method (str or None): Specifies the training method to be applied (e.g., 'CV200x', 'SMOTE', or None for default).

        Effects:
        - Updates the internal dictionary of models to include the new model along with its configuration.

        Notes:
        - If 'herder' is specified as the model_name, the model configuration links directly to a predefined HerderModel instance.
        """
        if model_name != "herder":
            self.models[model_name] = {
                "features": features,
                "results": {},
                "estimator": LogisticRegression(solver='liblinear', random_state=42),
                "use_mcp_scores": use_mcp_scores,
                "train_method": train_method
            }
        else:
            # Handle Herder model differently
            self.models[model_name] = {
                "features": features,
                "results": {},
                "custom_model": self.herder_model,  # Direct reference to the HerderModel instance
                "use_mcp_scores": use_mcp_scores,
                "train_method": train_method
            }

    def reset_models(self):
        """Clears the models dictionary to remove old models and their results."""
        print("Resetting models...")
        self.models = {}

    def train_models(self):
        """
        Trains various models specified in the `models` attribute of the class instance. This function handles
        different training configurations and methodologies including cross-validation and SMOTE for handling imbalanced data.

        The function iteratively processes each model configuration, prepares the data by optionally including
        MCP (Model Confidence Prediction) scores, selects relevant features, and applies the appropriate training method
        based on the model configuration.

        For the 'herder' model, a custom training procedure is invoked due to the 2 step process. For other models, standard or SMOTE-enhanced
        cross-validation techniques are used based on the configuration.

        Returns:
        - trained_models (dict): A dictionary containing the trained model instances, keyed by model name.

        Side Effects:
        - Prints the training progress and, for the 'herder' model, the model formulae after training.
        - Modifies the `models` attribute by updating the estimator objects with the trained instances.
        """

        trained_models = {}
        for model_name, model_info in self.models.items():
            print(f"\nTraining {model_name} model...")
            X, y = self.preprocessor.load_and_transform_data(model_name=model_name)
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
                if model_info["train_method"]=="CV200x":
                    self.train_with_200x_cross_validation(X_selected, y, estimator, model_name)
                elif model_info["train_method"]=="SMOTE":
                    self.train_with_smote_cross_validation(X_selected, y, estimator, model_name)
                else:
                    self.train_with_cross_validation(X_selected, y, estimator, model_name)
                model_info["estimator"] = estimator
                trained_models[model_name] = estimator
            else:
                # For the Herder model, call its fit_and_evaluate method directly
                self.herder_model.fit(X, y)
                self.herder_model.print_model_formulae()
                trained_models[model_name] = model_info["custom_model"]

        return trained_models

    def train_with_200x_cross_validation(self, X, y, estimator, model_name, n_splits=5, n_iterations=25, desired_percentage=0.95):
        """
        Trains the specified estimator using 200x cross-validation, which involves multiple iterations of Stratified K-Fold.
        This method aims to stabilize the estimation of model performance by repeatedly sampling and training.

        Parameters:
        - X (DataFrame): Feature matrix.
        - y (array-like): Target vector.
        - estimator (estimator instance): An instance of a scikit-learn estimator.
        - model_name (str): Name of the model, used for referencing in outputs.
        - n_splits (int): Number of splits for the Stratified K-Fold.
        - n_iterations (int): Number of cross-validation iterations.
        - desired_percentage (float): Target percentage for custom threshold determination.

        Effects:
        - Trains the model multiple times across specified iterations and folds, calculates metrics,
          and aggregates them to provide robust performance estimates.
        """
        all_iteration_metrics = []  # List to hold aggregated metrics for each iteration

        for iteration in range(0, n_iterations):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + iteration)
            iteration_metrics = []  # List to hold metrics for each fold in the current iteration

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                estimator.fit(X_train, y_train)
                y_test_proba = estimator.predict_proba(X_test)[:, 1]

                custom_threshold = self.determine_custom_threshold(y_test, y_test_proba, desired_percentage=desired_percentage)
                y_test_pred_custom = (y_test_proba >= custom_threshold).astype(int)

                metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred_custom),
                    'precision': precision_score(y_test, y_test_pred_custom, zero_division=0),
                    'recall': recall_score(y_test, y_test_pred_custom),
                    'f1_score': f1_score(y_test, y_test_pred_custom),
                    'roc_auc': roc_auc_score(y_test, y_test_proba),
                }

                iteration_metrics.append(metrics)
            aggregated_iteration_metrics = self.aggregate_metrics(iteration_metrics)
            all_iteration_metrics.append(aggregated_iteration_metrics)
        final_aggregated_metrics = self.aggregate_metrics_across_iterations(all_iteration_metrics)
        self.models[model_name] = {'aggregated_metrics': final_aggregated_metrics}
        self.print_aggregated_metrics(f"After {n_iterations} Iterations of 5-Fold Cross-Validation", final_aggregated_metrics)

    def train_with_smote_cross_validation(self, X, y, estimator, model_name, n_splits=5, desired_percentage=1):
        """
        Trains the specified estimator using SMOTE (Synthetic Minority Over-sampling Technique) and cross-validation.
        This method is particularly used to handle class imbalance by generating synthetic samples.

        Parameters:
        - X (DataFrame): Feature matrix.
        - y (array-like): Target vector.
        - estimator (estimator instance): An instance of a scikit-learn estimator.
        - model_name (str): Name of the model, used for referencing in outputs.
        - n_splits (int): Number of splits for the Stratified K-Fold.
        - desired_percentage (float): Target percentage for custom threshold determination which impacts metric calculation.

        Effects:
        - Applies SMOTE to balance the dataset, trains the estimator on the augmented data, and evaluates it using
          cross-validation to calculate various performance metrics.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []  # List to hold metrics for each fold
        probabilities = []
        y_tests = []
        custom_thresholds = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Initialize SMOTE and resample the training set
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Clone the estimator for each fold to ensure independence
            cloned_estimator = clone(estimator)
            cloned_estimator.fit(X_train_resampled, y_train_resampled)

            # Make predictions on the test set
            y_train_proba = cloned_estimator.predict_proba(X_train_resampled)[:, 1]
            y_test_proba = cloned_estimator.predict_proba(X_test)[:, 1]
            probabilities.extend(y_test_proba.tolist())
            y_tests.extend(y_test.tolist())
            # Metrics before applying custom threshold
            y_test_pred = (y_test_proba >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
            metrics_before = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, zero_division=0),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'specificity': tn / (tn + fp)
            }

            # Determine and apply custom threshold
            custom_threshold = self.determine_custom_threshold(y_test, y_test_proba, metric=self.threshold_metric, desired_percentage=desired_percentage)
            custom_thresholds.append(custom_threshold)
            y_test_pred_custom = (y_test_proba >= custom_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_custom).ravel()
            # Metrics after applying custom threshold
            metrics_after = {
                'accuracy': accuracy_score(y_test, y_test_pred_custom),
                'precision': precision_score(y_test, y_test_pred_custom, zero_division=0),
                'recall': recall_score(y_test, y_test_pred_custom),
                'f1_score': f1_score(y_test, y_test_pred_custom),
                'specificity': tn / (tn + fp)
            }

            # Calculate standard metrics and ROC curve data
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
            train_roc_auc = roc_auc_score(y_train_resampled, y_train_proba)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            train_fpr, train_tpr, _ = roc_curve(y_train_resampled, y_train_proba)
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

    def train_with_cross_validation(self, X, y, estimator, model_name, n_splits=5, desired_percentage=0.95):
        """
        Trains the specified estimator using standard cross-validation.

        Parameters:
        - X (DataFrame): Feature matrix.
        - y (array-like): Target vector.
        - estimator (estimator instance): An instance of a scikit-learn estimator.
        - model_name (str): Name of the model, used for referencing in outputs.
        - n_splits (int): Number of splits for the Stratified K-Fold.
        - desired_percentage (float): Target percentage for custom threshold determination, affecting metrics like specificity.

        Effects:
        - Trains the estimator using cross-validation, computes metrics at a default and custom threshold,
          and aggregates these metrics to assess model performance.
        """
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
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
            metrics_before = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, zero_division=0),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'specificity': tn / (tn + fp)
            }

            # Determine and apply custom threshold
            custom_threshold = self.determine_custom_threshold(y_test, y_test_proba, metric=self.threshold_metric, desired_percentage=desired_percentage)
            custom_thresholds.append(custom_threshold)
            y_test_pred_custom = (y_test_proba >= custom_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_custom).ravel()
            # Metrics after applying custom threshold
            metrics_after = {
                'accuracy': accuracy_score(y_test, y_test_pred_custom),
                'precision': precision_score(y_test, y_test_pred_custom, zero_division=0),
                'recall': recall_score(y_test, y_test_pred_custom),
                'f1_score': f1_score(y_test, y_test_pred_custom),
                'specificity': tn / (tn + fp)
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
        self.calculate_and_store_metrics(model_name, fold_metrics)


    def aggregate_metrics(self, iteration_metrics):
        aggregated = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
        for fm in iteration_metrics:
            for metric in aggregated:
                aggregated[metric].append(fm[metric])
        return {metric: {'avg': np.mean(values), 'std': np.std(values)} for metric, values in aggregated.items()}

    def aggregate_metrics_across_iterations(self, all_iteration_metrics):
        aggregated = {metric: {'avg': [], 'std': []} for metric in all_iteration_metrics[0]}
        for iteration in all_iteration_metrics:
            for metric in aggregated:
                aggregated[metric]['avg'].append(iteration[metric]['avg'])
                aggregated[metric]['std'].append(iteration[metric]['std'])
        return {metric: {'avg': np.mean(values['avg']), 'std': np.mean(values['std'])} for metric, values in aggregated.items()}

    def print_aggregated_metrics(self, context, metrics):
        print(f"\nAggregated Metrics {context}:")
        for metric, stats in metrics.items():
            print(f"{metric.capitalize()}: Avg = {stats['avg']:.4f} (±{stats['std']:.4f})")

    def determine_custom_threshold(self, y, y_proba, metric='ppv', desired_percentage=None):
        """
        Determines a custom threshold for classification based on the desired metric's target percentage.

        This function calculates thresholds using either positive predictive value (PPV) or negative predictive value (NPV)
        to determine the smallest threshold which meets or exceeds the desired percentage of the specified metric.

        Parameters:
        - y (array-like): True binary labels.
        - y_proba (array-like): Probabilities of the positive class.
        - metric (str): Metric to use for thresholding ('ppv' for positive predictive value, 'npv' for negative predictive value).
        - desired_percentage (float): The minimum desired percentage for the chosen metric.

        Returns:
        - float: The determined threshold that meets or exceeds the desired metric percentage, defaults to 0.5 if no such threshold is found.

        Raises:
        - ValueError: If an invalid metric is specified.
        """
        if metric == 'ppv':
            precision, recall, thresholds = precision_recall_curve(y, y_proba)
            eligible_indices = np.where(precision[:-1] >= desired_percentage)[0]
            if eligible_indices.size > 0:
                threshold_index = eligible_indices[0]  # Use the first index where precision is above the desired percentage
                ppv_threshold = thresholds[threshold_index]
            else:
                # Default threshold if no precision value meets the desired percentage
                ppv_threshold = 0.5
            return ppv_threshold

        elif metric == 'npv':
            y_np = y.values if hasattr(y, 'values') else y

            # Sort the probabilities and corresponding true labels
            sorted_indices = np.argsort(y_proba)
            sorted_proba = y_proba[sorted_indices]
            sorted_y_np = y_np[sorted_indices]

            # Initialize variables to track the best threshold and its NPV
            best_threshold = None
            best_npv = 0

            for idx, threshold in enumerate(sorted_proba[:-1]):  # Exclude the last one to prevent division by zero
                # Predictions based on current threshold
                y_pred = (sorted_proba > threshold).astype(int)

                # Confusion matrix elements
                tn, fp, fn, tp = confusion_matrix(sorted_y_np, y_pred).ravel()

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
        """
        Calculates the Model Confidence Prediction (MCP) scores for the input features based on a logistic regression model.

        This function computes a logistic regression prediction without using the logistic regression model directly.
        Instead, it uses the coefficients and intercepts manually for the calculation.

        Parameters:
        - X (DataFrame): The input features for which MCP scores are to be calculated.

        Returns:
        - Series: A series containing the MCP scores for the input data.

        Outputs:
        - Prints the maximum MCP score encountered during the calculations.
        """
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
        """
        Aggregates and calculates performance metrics from cross-validation folds and stores them in the model's dictionary.

        This function takes the metrics from individual folds, calculates their average and standard deviation, and
        aggregates ROC data. It then prints and stores these aggregated metrics.

        Parameters:
        - model_name (str): The name of the model for which metrics are being calculated and stored.
        - fold_metrics (list of dicts): A list containing metric dictionaries for each fold, which include metrics both before
          and after threshold adjustment, as well as ROC curve data.

        Effects:
        - Updates the model's entry in `self.models` with aggregated metrics and ROC data.
        - Prints aggregated metrics and their standard deviations.
        """
        # Initialize containers for aggregated metrics
        aggregated_metrics_before = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'specificity': []}
        aggregated_metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'specificity': []}
        aggregated_roc_data = {
            'train_fpr': [], 'train_tpr': [], 'train_roc_auc': [],
            'test_fpr': [], 'test_tpr': [], 'test_roc_auc': []
        }

        # Loop through each fold to aggregate metrics
        for fm in fold_metrics:
            for metric in aggregated_metrics_before:
                # Check if metrics are present before aggregating
                if metric in fm['metrics_before']:
                    aggregated_metrics_before[metric].append(fm['metrics_before'][metric])
                if metric in fm['metrics_after']:
                    aggregated_metrics_after[metric].append(fm['metrics_after'][metric])

            for roc_key in ['train_fpr', 'train_tpr', 'train_roc_auc', 'test_fpr', 'test_tpr', 'test_roc_auc']:
                if roc_key in fm:
                    aggregated_roc_data[roc_key].append(fm[roc_key])

        # Calculate averages and standard deviations for metrics
        final_metrics_before = {metric: {'avg': np.mean(values), 'std': np.std(values)} for metric, values in aggregated_metrics_before.items()}
        final_metrics_after = {metric: {'avg': np.mean(values), 'std': np.std(values)} for metric, values in aggregated_metrics_after.items()}
        avg_train_roc_auc = np.mean(aggregated_roc_data['train_roc_auc'])
        avg_test_roc_auc = np.mean(aggregated_roc_data['test_roc_auc'])

        # Store aggregated metrics in the model's dictionary
        self.models[model_name]['metrics_waybefore'] = aggregated_metrics_before
        self.models[model_name]['metrics_beforeafter'] = aggregated_metrics_after
        self.models[model_name]['metrics_before'] = final_metrics_before
        self.models[model_name]['metrics_after'] = final_metrics_after
        self.models[model_name]['roc_data'] = aggregated_roc_data

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


    def perform_wilcoxon_signed_rank_test(self):
        """Compare each model's aggregated metrics against BTS guideline metrics using Wilcoxon signed-rank test."""
        if not self.guideline_data:
            print("Guideline metrics not properly provided.")
            return

        bts_metrics = self.guideline_data
        print("\nComparing model metrics against BTS guidelines:")

        for model_name, model_info in self.models.items():
            print(f"\nResults for {model_name}:")
            if 'metrics_before' not in model_info or 'metrics_after' not in model_info:
                print("Metrics not properly stored for this model.")
                continue

            # Get metrics before and after threshold adjustment
            metrics_before = model_info['metrics_waybefore']
            metrics_after = model_info['metrics_beforeafter']

            # Extract values for each metric
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                if metric not in metrics_before or metric not in metrics_after:
                    print(f"Metric {metric} not found for this model.")
                    continue

                model_metric_values_before = metrics_before[metric]
                model_metric_values_after = metrics_after[metric]
                bts_metric_values = [entry[metric] for entry in bts_metrics]

                # Ensure the number of observations is the same for Wilcoxon test
                if len(model_metric_values_before) != len(bts_metric_values):
                    print(model_metric_values_before)
                    print(len(bts_metric_values))
                    print(f"Cannot perform Wilcoxon test for {metric} due to unequal sample sizes.")
                    continue

                # Perform Wilcoxon Signed-Rank Test
                try:
                    stat_before, p_value_before = stats.wilcoxon(model_metric_values_before, bts_metric_values, zero_method='wilcox', alternative='two-sided')
                    print(f"{metric.capitalize()} Comparison Before: W-stat={stat_before}, P-value={p_value_before:.4f}")
                    # Repeat for 'after' metrics if applicable
                    stat_after, p_value_after = stats.wilcoxon(model_metric_values_after, bts_metric_values, zero_method='wilcox', alternative='two-sided')
                    print(f"{metric.capitalize()} Comparison After: W-stat={stat_after}, P-value={p_value_after:.4f}")
                except ValueError as e:
                    print(f"Error performing Wilcoxon test for {metric}: {str(e)}")

    def perform_mann_whitney_test(self):
        """Compare each model's aggregated metrics against BTS guideline metrics using Mann-Whitney U test."""
        if not self.guideline_data:
            print("Guideline metrics not properly provided.")
            return

        bts_metrics = self.guideline_data
        print("\nComparing model metrics against BTS guidelines:")

        for model_name, model_info in self.models.items():
            print(f"\nResults for {model_name}:")
            if 'metrics_before' not in model_info or 'metrics_after' not in model_info:
                print("Metrics not properly stored for this model.")
                continue

            # Get metrics before and after threshold adjustment
            metrics_before = model_info['metrics_waybefore']
            metrics_after = model_info['metrics_beforeafter']

            # Extract values for each metric
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                if metric not in metrics_before or metric not in metrics_after:
                    print(f"Metric {metric} not found for this model.")
                    continue

                model_metric_values_before = metrics_before[metric]
                model_metric_values_after = metrics_after[metric]
                bts_metric_values = [entry[metric] for entry in bts_metrics]

                # Perform Mann-Whitney U Test
                u_stat_before, p_value_before = stats.mannwhitneyu(model_metric_values_before, bts_metric_values, alternative='two-sided')
                u_stat_after, p_value_after = stats.mannwhitneyu(model_metric_values_after, bts_metric_values, alternative='two-sided')
                print(f"{metric.capitalize()} Comparison Before: U-stat={u_stat_before}, P-value={p_value_before:.4f}")
                # print(f"{metric.capitalize()} Comparison After: U-stat={u_stat_after}, P-value={p_value_after:.4f}")

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
        X, y = self.preprocessor.load_and_transform_data(model_name)

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
        """
        Compiles a unique list of all features used across various models within the ensemble, including any
        Model Confidence Prediction (MCP) features if they are utilized by any model.

        This function iterates over each model configuration, collects standard features, and conditionally includes
        MCP-related features based on model settings.

        Returns:
        - list: A list of unique features used across the ensemble models.
        """
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
        sns.histplot(probabilities[true_labels == 0], bins=20, kde=True, color='blue', alpha=0.5, label='No LC')
        sns.histplot(probabilities[true_labels == 1], bins=20, kde=True, color='red', alpha=0.7, label='NSCLC')

        plt.axvline(x=mean_threshold, color='green', linestyle='--', label=f'Mean Threshold: {mean_threshold:.2f}')
        # plt.axvline(x=0.5, color='green', linestyle='--', label=f'Threshold: 0.5')
        plt.title(f'Prediction Probability Distribution for {model_name} Model', fontsize=10)
        plt.xlabel('Predicted Probability of Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

    def plot_roc_curves(self, model_name, curve_type='test'):
        """
        Plots the Receiver Operating Characteristic (ROC) curves

        Parameters:
        - model_name (str): The name of the model for which to plot the ROC curves.
        - curve_type (str): Type of curve to plot ('train' or 'test'), default is 'test'.

        Outputs:
        - A plot displaying the ROC curves for all folds, the mean ROC curve, and the chance line.
        """

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

            ax.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean {curve_type.capitalize()} ROC (AUC = {mean_auc:.2f} ± {std_auc:.4f})', lw=2, alpha=0.8)
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


        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm_default, annot=True, fmt="d", cmap='Blues')
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title(f'Confusion Matrix {model_name} (Threshold 0.5)\nSensitivity: {sensitivity_default:.2f}, Specificity: {specificity_default:.2f}', fontsize=10)
        # plt.xticks(ticks=np.arange(2) + 0.5, labels=['Negative', 'Positive'], fontsize=10)
        # plt.yticks(ticks=np.arange(2) + 0.5, labels=['Negative', 'Positive'], rotation=0, fontsize=10)

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

