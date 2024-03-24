import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor
from herder_model import HerderModel


class Model:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.models = {}
        self.n_estimators = 10
        self.herder_model = HerderModel(self.preprocessor)
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
                "results": [],
                "estimator": LogisticRegression(solver='liblinear', random_state=42),
                "use_mcp_scores": use_mcp_scores
            }
        else:
            # Handle Herder model differently
            self.models[model_name] = {
                "features": features,
                "results": [],
                "custom_model": self.herder_model  # Direct reference to the HerderModel instance
            }

    def reset_models(self):
        """Clears the models dictionary to remove old models and their results."""
        print("Resetting models...")
        self.models = {}

#================================================#
# REVERSE COMMENT FOR 200*5-FOLD CROSSVALIDATION #
#================================================#

#     def train_models(self):
        # X, y = self.preprocessor.load_and_transform_data()
        # trained_models = {}

        # for model_name, model_info in self.models.items():
            # print(f"\nTraining {model_name} model...")
            # X_mod = X.copy()
            # if model_info["use_mcp_scores"]:
                # # Calculate MCP_score and add it to the DataFrame before training
                # X_mod['MCP_score'] = self.calculate_mcp_scores(X_mod)

            # X_selected = X_mod[model_info["features"]]

            # if model_info["use_mcp_scores"] and 'MCP_score' not in model_info["features"]:
                # X_selected = X_mod[model_info["features"] + ['MCP_score']]

            # if model_name != "herder":
                # estimator = clone(model_info["estimator"])
                # self.train_with_cross_validation(X_selected, y, estimator, model_name)
                # model_info["estimator"] = estimator
                # trained_models[model_name] = estimator
            # else:
                # self.herder_model.fit(X, y)
                # self.herder_model.print_model_formulae()
                # trained_models[model_name] = model_info["custom_model"]

            # # Print the aggregated metrics for each model after training
            # if 'aggregated_metrics' in self.models[model_name]:
                # print(f"\nAggregated Metrics for {model_name}:")
                # for metric, value in self.models[model_name]['aggregated_metrics'].items():
                    # print(f"{metric}: {value:.4f}")

        # return trained_models

    # def train_with_cross_validation(self, X, y, estimator, model_name, n_splits=5, n_iterations=200):
        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # # Initialize storage for results across all iterations
        # all_iterations_results = []

        # for iteration in range(n_iterations):
            # print(f"Iteration {iteration + 1}/{n_iterations}")
            # iteration_results = {
                # 'fold_metrics': [],
                # 'y_test_all': [],
                # 'y_proba_all': []
            # }

            # for train_index, test_index in skf.split(X, y):
                # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                # y_train, y_test = y[train_index], y[test_index]

                # estimator.fit(X_train, y_train)

                # y_train_pred = estimator.predict(X_train)
                # y_train_proba = estimator.predict_proba(X_train)[:, 1] if hasattr(estimator, "predict_proba") else []

                # y_test_pred = estimator.predict(X_test)
                # y_test_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else []

                # # Calculate and store metrics for this fold
                # fold_metric = self.calculate_fold_metrics(y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba)
                # iteration_results['fold_metrics'].append(fold_metric)

                # # Store actual and predicted results for each fold
                # iteration_results['y_test_all'].extend(y_test.tolist())
                # iteration_results['y_proba_all'].extend(y_test_proba.tolist())

            # # After finishing all folds for the current iteration, store iteration results
            # all_iterations_results.append(iteration_results)

        # # Aggregated metrics across iterations are calculated here if necessary
        # # For simplicity, we'll just store all_iterations_results directly
        # self.models[model_name]['aggregated_results'] = all_iterations_results

        # # Train the model on the entire dataset after cross-validation
        # estimator.fit(X, y)
        # self.models[model_name]['estimator'] = estimator

    # def calculate_fold_metrics(self, y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba):
        # # Existing metrics calculations...
        # metrics = {
            # "accuracy": accuracy_score(y_test, y_test_pred),
            # "precision": precision_score(y_test, y_test_pred, zero_division=0),
            # "recall": recall_score(y_test, y_test_pred),
            # "f1": f1_score(y_test, y_test_pred),
            # "train_roc_auc": roc_auc_score(y_train, y_train_proba) if len(y_train_proba) > 0 else None,
            # "val_roc_auc": roc_auc_score(y_test, y_test_proba) if len(y_test_proba) > 0 else None,
            # "y_test": y_test.tolist(),  # Ensure these are lists for easy JSON serialization if needed
            # "y_test_proba": y_test_proba.tolist(),
        # }
        # return metrics


    # def aggregate_fold_metrics(self, fold_metrics):
        # aggregated = {}
        # for metric in fold_metrics[0].keys():
            # aggregated[metric] = np.mean([fm[metric] for fm in fold_metrics])
        # return aggregated
    # def aggregate_all_iterations_metrics(self, all_fold_metrics):
        # # Initialize a dictionary to hold aggregated metrics across iterations
        # aggregated = {metric: [] for metric in all_fold_metrics[0].keys()}

        # # Loop through each metric and aggregate across iterations
        # for iteration_metrics in all_fold_metrics:
            # for metric, value in iteration_metrics.items():
                # aggregated[metric].append(value)

        # # Calculate mean and standard deviation for each metric across iterations
        # final_aggregated = {}
        # for metric, values in aggregated.items():
            # final_aggregated[metric + "_mean"] = np.mean(values)
            # final_aggregated[metric + "_std"] = np.std(values)

        # return final_aggregated


#     def plot_prediction_histograms(self, model_name):
        # if model_name not in self.models or 'aggregated_results' not in self.models[model_name]:
            # print(f"No aggregated results found for the model '{model_name}'.")
            # return

        # # Initialize a dictionary to hold probabilities for each instance across all iterations
        # probability_sums = {}
        # count_per_instance = {}

        # # Iterate over all stored iterations to sum probabilities and count occurrences for each instance
        # for iteration_results in self.models[model_name]['aggregated_results']:
            # for i, prob in enumerate(iteration_results['y_proba_all']):
                # if i not in probability_sums:
                    # probability_sums[i] = 0
                    # count_per_instance[i] = 0
                # probability_sums[i] += prob
                # count_per_instance[i] += 1

        # # Calculate mean probabilities
        # mean_probabilities = [probability_sums[i] / count_per_instance[i] for i in sorted(probability_sums.keys())]
        # all_true_labels = [iteration_results['y_test_all'][i] for i in sorted(count_per_instance.keys())]

        # # Convert lists to numpy arrays for plotting
        # mean_probabilities = np.array(mean_probabilities)
        # all_true_labels = np.array(all_true_labels)

        # # Plot the histograms for both classes
        # plt.figure(figsize=(10, 6))
        # sns.histplot(mean_probabilities[all_true_labels == 0], bins=20, stat="density", kde=True, color='blue', alpha=0.5, label='Negative Class')
        # sns.histplot(mean_probabilities[all_true_labels == 1], bins=20, stat="density", kde=True, color='red', alpha=0.7, label='Positive Class')

        # plt.axvline(x=0.5, color='green', linestyle='--', label='Decision Threshold (0.5)')
        # plt.title(f'Mean Prediction Probability Distribution for {model_name}', fontsize=20)
        # plt.xlabel('Mean Predicted Probability of Positive Class', fontsize=16)
        # plt.ylabel('Density', fontsize=16)
        # plt.legend(fontsize=12)
        # plt.xlim(0, 1)
        # plt.grid(True)
        # plt.show()

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

    def plot_prediction_histograms(self, model_name):
        if model_name not in self.models:
            print(f"No model found with the name {model_name}")
            return

        if 'results' not in self.models[model_name] or not self.models[model_name]['results']:
            print(f"No predictions available for the {model_name} model")
            return

        plt.figure(figsize=(15, 7))

        true_labels = np.concatenate([result[0] for result in self.models[model_name]["results"]])
        predictions = np.concatenate([result[2] for result in self.models[model_name]["results"]])

        # Plot histograms for true negative and true positive predictions
        sns.histplot(predictions[true_labels == 0], bins=20, stat="density", kde=True, color='blue', alpha=0.5, label='Negative Class')
        sns.histplot(predictions[true_labels == 1], bins=20, stat="density", kde=True, color='red', alpha=0.7, label='Positive Class')

        plt.title(f'Prediction Probability Distribution for {model_name} Model', fontsize=20)
        plt.xlabel('Predicted Probability of Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

#=======================================================#
# END OF REVERSE COMMENT FOR 200*5-fold crossvalidation #
#=======================================================#

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
        # Initialize lists to collect metrics
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        train_roc_aucs, val_roc_aucs = [], []

        # ROC data initialization
        roc_data = {'train': {'fpr': [], 'tpr': [], 'roc_auc': []},
                    'validation': {'fpr': [], 'tpr': [], 'roc_auc': []}}

        for metrics in fold_metrics:
            # Unpack metrics
            y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba, \
            train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc = metrics

            # Calculate metrics
            accuracies.append(accuracy_score(y_test, y_test_pred))
            precisions.append(precision_score(y_test, y_test_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_test_pred))
            f1_scores.append(f1_score(y_test, y_test_pred))

            # Store train and validation ROC AUC for metrics calculation
            train_roc_aucs.append(train_roc_auc)
            val_roc_aucs.append(val_roc_auc)

            # Store ROC curve data
            roc_data['train']['fpr'].append(train_fpr)
            roc_data['train']['tpr'].append(train_tpr)
            roc_data['train']['roc_auc'].append(train_roc_auc)
            roc_data['validation']['fpr'].append(val_fpr)
            roc_data['validation']['tpr'].append(val_tpr)
            roc_data['validation']['roc_auc'].append(val_roc_auc)

        # Aggregate and store metrics
        self.models[model_name]['metrics'] = {
            "Accuracy": np.mean(accuracies),
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "F1": np.mean(f1_scores),
            "Train ROC AUC": np.mean(train_roc_aucs),
            "Validation ROC AUC": np.mean(val_roc_aucs),
        }
        self.models[model_name]['roc_data'] = roc_data
        # Calculate standard deviations for metrics
        std_metrics = {
            "Accuracy STD": np.std(accuracies),
            "Precision STD": np.std(precisions),
            "Recall STD": np.std(recalls),
            "F1 STD": np.std(f1_scores),
            "Train ROC AUC STD": np.std(train_roc_aucs),
            "Validation ROC AUC STD": np.std(val_roc_aucs)
        }

        # Store the standard deviation metrics
        self.models[model_name]['metrics_std'] = std_metrics

        # Print summary of metrics including Train and Validation ROC AUC under each threshold section
        print(f"Metrics for {model_name} Model:")
        for metric, value in self.models[model_name]['metrics'].items():
            std_metric = self.models[model_name]['metrics_std'][f"{metric} STD"]
            print(f"{metric}: {value:.4f} (std: {std_metric:.4f})")

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
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean {label_prefix} ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)
        ax.fill_between(mean_fpr, mean_tpr - np.std(tprs, axis=0), mean_tpr + np.std(tprs, axis=0), color='grey', alpha=0.2, label=f'{label_prefix} ± 1 std. dev.')

        ax.legend(loc="lower right")

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
