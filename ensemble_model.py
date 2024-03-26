import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor


class VotingModel:
    def __init__(self, trained_models, ensemble_features, filepath, target, binary_map, model_name, threshold_metric='ppv'):
        self.data_preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.trained_models = trained_models
        self.ensemble_features = ensemble_features
        self.voting_classifier = None
        self.results = {}
        self.X = None
        self.y = None
        self.threshold_metric = threshold_metric
        self.load_data()
        self.model_name = model_name

    def reset(self):
        self.voting_classifier = None
        self.results = {}

    def load_data(self):
        self.X, self.y = self.data_preprocessor.load_and_transform_data()
        self.X = self.X[self.ensemble_features]

    def train_voting_classifier(self, desired_percentage=0.95):
        self.reset()
        # Correcting the way estimators are collected based on the new trained_models structure
        estimators = [(name, model) for name, model in self.trained_models.items()]
        self.voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metrics_before = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
        metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
        thresholds = []
        aggregated_probabilities = []
        aggregated_y_test = []

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.voting_classifier.fit(X_train, y_train)

            y_train_proba = self.voting_classifier.predict_proba(X_train)[:, 1]
            y_proba = self.voting_classifier.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            aggregated_probabilities.extend(y_proba.tolist())
            aggregated_y_test.extend(y_test.tolist())

            # Metrics calculation remains unchanged
            self.append_metrics(metrics_before, y_train, y_train_proba, y_test, y_proba, y_pred)
            threshold = self.determine_custom_threshold(y_test, y_proba, desired_percentage)
            thresholds.append(threshold)

            y_pred_adjusted = (y_proba >= threshold).astype(int)
            self.append_metrics(metrics_after, y_train, y_train_proba, y_test, y_proba, y_pred_adjusted)

        # Finalizing results and printing metrics remains unchanged
        self.results['probabilities'] = np.array(aggregated_probabilities)
        self.results['y_test'] = np.array(aggregated_y_test)
        self.results['thresholds'] = thresholds
        self.results['average_threshold'] = np.mean(thresholds)
        self.print_metrics("(before custom threshold)", metrics_before)
        self.print_metrics("(after custom threshold)", metrics_after)

    def determine_custom_threshold(self, y_test, y_proba, desired_percentage):
        if self.threshold_metric == 'ppv':
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
            threshold_indices = np.where(precision >= desired_percentage)[0]
            return thresholds_pr[threshold_indices[0] - 1] if threshold_indices.size > 0 else 1.0
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

    def append_metrics(self, metrics_dict, y_train, y_train_proba, y_test, y_proba, y_pred):
        metrics_dict['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics_dict['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics_dict['recall'].append(recall_score(y_test, y_pred))
        metrics_dict['f1'].append(f1_score(y_test, y_pred))
        metrics_dict['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
        metrics_dict['val_roc_auc'].append(roc_auc_score(y_test, y_proba))


    def print_metrics(self, phase, metrics):
        print(f"\nmetrics for the {self.model_name} model {phase}:")
        for metric_name, values in metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric_name.capitalize()}: {avg_value:.4f} (std: {std_value:.4f})")

    def plot_roc_curves(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 8))

        # Assuming self.results['average_threshold'] stores the mean threshold across folds
        avg_threshold = self.results['average_threshold']

        for i, (train, test) in enumerate(cv.split(self.X, self.y)):
            self.voting_classifier.fit(self.X.iloc[train], self.y.iloc[train])
            y_proba = self.voting_classifier.predict_proba(self.X.iloc[test])
            fpr, tpr, thresholds = roc_curve(self.y.iloc[test], y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Interpolate TPR for the average threshold
        interp_mean_tpr_at_threshold = np.interp(avg_threshold, mean_fpr, mean_tpr)

        # Plot the mean ROC curve
        plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

        # Plot the average threshold point
        plt.scatter(avg_threshold, interp_mean_tpr_at_threshold, color='black', label=f'Avg. {self.threshold_metric} Threshold at FPR={avg_threshold:.2f}, TPR={interp_mean_tpr_at_threshold:.2f}')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve for {self.model_name} Model')
        plt.legend(loc="lower right")
        plt.show()

    def generate_shap_plot(self):
        if not self.voting_classifier:
            print("Voting classifier is not trained.")
            return

        # SHAP Explainer initialization may vary based on the model type
        explainer = shap.Explainer(self.voting_classifier.predict, self.X)
        shap_values = explainer(self.X)

        # Plotting
        shap.plots.bar(shap_values, max_display=30)
        shap.plots.beeswarm(shap_values, max_display=30, show=False)
        plt.title(f"{self.model_name} model SHAP Beeswarm Plot", fontsize=20)
        plt.show()

    def plot_prediction_histograms(self):
        if not self.results or 'y_test' not in self.results or self.results['probabilities'] is None:
            print("No results, necessary data, or probabilities found for the voting model.")
            return

        probabilities = self.results['probabilities']
        true_labels = self.results['y_test']
        average_threshold = self.results.get('average_threshold', 0.5)

        plt.figure(figsize=(15, 7))
        sns.histplot(probabilities[true_labels == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(probabilities[true_labels == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        if average_threshold is not None:
            plt.axvline(x=average_threshold, color='green', linestyle='--', label=f'{self.threshold_metric} Threshold: {average_threshold:.2f}')

        plt.title(f'Probability Distribution for the {self.model_name} Model', fontsize=10)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.show()

    def plot_confusion_matrices(self):
        if not self.results or 'y_test' not in self.results or 'probabilities' not in self.results:
            print("No results, necessary data, or probabilities found for the voting model.")
            return

        # Retrieve true labels and predicted probabilities
        true_labels = self.results['y_test']
        probabilities = self.results['probabilities']
        average_threshold = self.results.get('average_threshold', 0.5)

        # Convert probabilities to binary predictions using the default and the custom average thresholds
        default_predictions = (probabilities >= 0.5).astype(int)
        custom_predictions = (probabilities >= average_threshold).astype(int)

        # Compute confusion matrices for both default and custom threshold predictions
        cm_default = confusion_matrix(true_labels, default_predictions)
        cm_custom = confusion_matrix(true_labels, custom_predictions)
                # For the default threshold
        TP_default = cm_default[1, 1]
        FN_default = cm_default[1, 0]
        FP_default = cm_default[0, 1]
        TN_default = cm_default[0, 0]

        sensitivity_default = TP_default / (TP_default + FN_default)
        specificity_default = TN_default / (TN_default + FP_default)

        # For the custom threshold
        TP_custom = cm_custom[1, 1]
        FN_custom = cm_custom[1, 0]
        FP_custom = cm_custom[0, 1]
        TN_custom = cm_custom[0, 0]

        sensitivity_custom = TP_custom / (TP_custom + FN_custom)
        specificity_custom = TN_custom / (TN_custom + FP_custom)


        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        # Plotting the confusion matrix for the default threshold
        sns.heatmap(cm_default, annot=True, fmt=".2f", cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted labels')
        axes[0].set_ylabel('True labels')
        axes[0].set_title(f'Confusion Matrix {self.model_name} (Default Threshold 0.5)\nSensitivity: {sensitivity_default:.2f}, Specificity: {specificity_default:.2f}', fontsize=10)
        axes[0].set_xticklabels(['Negative', 'Positive'])
        axes[0].set_yticklabels(['Negative', 'Positive'], rotation=0)

        # Plotting the confusion matrix for the custom average threshold
        sns.heatmap(cm_custom, annot=True, fmt=".2f", cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted labels')
        axes[1].set_title(f'({self.threshold_metric} Threshold {average_threshold:.2f})\nSensitivity: {sensitivity_custom:.2f}, Specificity: {specificity_custom:.2f}', fontsize=10)
        axes[1].set_xticklabels(['Negative', 'Positive'])
        axes[1].set_yticklabels(['Negative', 'Positive'], rotation=0)

        plt.tight_layout()
        plt.show()

class score_based_ensemble:
    def __init__(self, filepath, target, binary_map, features, model_name, threshold_metric='ppv'):
        self.model = LogisticRegression(solver='liblinear')
        self.cv_scores = {}
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.score_ensemble_features = features
        self.model_name = model_name
        self.average_threshold = None
        self.threshold_metric=threshold_metric

    def fit_evaluate(self, n_splits=5, random_state=42, scoring=None, desired_percentage=0.95):
        X, y = self.preprocessor.load_and_transform_data()
        X_feature = X[self.score_ensemble_features]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        thresholds = []

        # Initialize dictionaries to store scores for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        pre_threshold_scores = {metric: [] for metric in metrics}
        post_threshold_scores = {metric: [] for metric in metrics}

        for train_idx, test_idx in cv.split(X_feature, y):
            self.model.fit(X_feature.iloc[train_idx], y[train_idx])
            y_proba = self.model.predict_proba(X_feature.iloc[test_idx])
            proba_positive_class = y_proba[:, 1] if y_proba.ndim == 2 else np.expand_dims(y_proba, axis=-1)

            # Calculate pre-threshold metrics
            y_pred = (proba_positive_class >= 0.5).astype(int)
            for metric in metrics:
                score = self._calculate_metric(metric, y[test_idx], y_pred, proba_positive_class)
                pre_threshold_scores[metric].append(score)

            # Determine threshold and calculate post-threshold metrics
            threshold = self._determine_threshold(desired_percentage, y[test_idx], proba_positive_class)
            thresholds.append(threshold)
            y_pred_threshold = (proba_positive_class >= threshold).astype(int)
            for metric in metrics:
                score = self._calculate_metric(metric, y[test_idx], y_pred_threshold, proba_positive_class)
                post_threshold_scores[metric].append(score)

        self.average_threshold = np.mean(thresholds)
        self.pre_cv_scores = {metric: np.mean(values) for metric, values in pre_threshold_scores.items()}
        self.post_cv_scores = {metric: np.mean(values) for metric, values in post_threshold_scores.items()}
        self.pre_cv_std = {metric: np.std(values) for metric, values in pre_threshold_scores.items()}
        self.post_cv_std = {metric: np.std(values) for metric, values in post_threshold_scores.items()}

    def _calculate_metric(self, metric, y_true, y_pred, y_proba):
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            return recall_score(y_true, y_pred)
        elif metric == 'f1':
            return f1_score(y_true, y_pred)
        elif metric == 'roc_auc':
            return roc_auc_score(y_true, y_proba)
        else:
            return None

    def _determine_threshold(self, desired_percentage, y_true, proba_positive_class):
        default_threshold = 0.5  # Define a sensible default threshold.

        if not isinstance(desired_percentage, (float, int)) or not (0 <= desired_percentage <= 1):
            raise ValueError("desired_percentage must be a number between 0 and 1.")

        if self.threshold_metric == 'ppv':
            precision, _, thresholds_pr = precision_recall_curve(y_true, proba_positive_class)
            target_indices = np.where(precision >= desired_percentage)[0]
            # Use the default_threshold if no target indices are found.
            return thresholds_pr[target_indices[0] - 1] if target_indices.size > 0 else default_threshold

        elif self.threshold_metric == 'npv':
            npvs = []
            threshold_values = np.linspace(0, 1, 100)
            for thresh in threshold_values:
                y_pred_threshold = (proba_positive_class >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_threshold).ravel()
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                npvs.append(npv)

            npvs = np.array(npvs)
            if not np.any(npvs >= desired_percentage):  # Check if any NPV meets the desired percentage.
                return default_threshold
            best_npv_index = np.argmin(np.abs(npvs - desired_percentage))
            return threshold_values[best_npv_index]

        else:
            raise ValueError("Invalid threshold_metric. Choose 'ppv' or 'npv'.")

    def plot_roc_curve(self, n_splits=5, random_state=42):
        X, y = self.preprocessor.load_and_transform_data()
        X_feature = X.loc[:, self.score_ensemble_features]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Assume average_threshold attribute exists
        avg_threshold = self.average_threshold  # Use the class attribute for average_threshold

        for fold, (train, test) in enumerate(cv.split(X_feature, y)):
            self.model.fit(X_feature.iloc[train], y[train])
            y_proba = self.model.predict_proba(X_feature.iloc[test])
            fpr, tpr, thresholds = roc_curve(y[test], y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold} (AUC = {roc_auc:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Interpolate TPR for the average threshold
        interp_mean_tpr_at_threshold = np.interp(avg_threshold, mean_fpr, mean_tpr)

        # Plot the mean ROC curve
        plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

        # Plot the average threshold point on the ROC curve
        plt.scatter(avg_threshold, interp_mean_tpr_at_threshold, color='black', zorder=5, label=f'Avg. Threshold at FPR={avg_threshold:.2f}, TPR={interp_mean_tpr_at_threshold:.2f}')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve for {self.model_name} Model')
        plt.legend(loc="lower right")
        plt.show()

    def plot_prediction_histogram(self):
        X, y = self.preprocessor.load_and_transform_data()
        X_ensemble = X.loc[:, self.score_ensemble_features]
        self.model.fit(X_ensemble, y)
        probabilities = self.model.predict_proba(X_ensemble)[:, 1]

        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 7))
        sns.histplot(probabilities[y == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(probabilities[y == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        # Plot average threshold
        plt.axvline(x=self.average_threshold, color='green', linestyle='--', label=f'Avg {self.threshold_metric} Threshold: {self.average_threshold:.2f}')

        plt.title(f'Probability Distribution for {self.model_name} with Avg {self.threshold_metric} Threshold {self.average_threshold:.2f}', fontsize=10)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
        plt.show()

    def plot_confusion_matrices(self, n_splits=5, random_state=42):
        X, y = self.preprocessor.load_and_transform_data()
        X_feature = X.loc[:, self.score_ensemble_features]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        aggregated_true_labels = []
        aggregated_probabilities = []

        for train, test in cv.split(X_feature, y):
            self.model.fit(X_feature.iloc[train], y[train])
            y_proba = self.model.predict_proba(X_feature.iloc[test])[:, 1]

            # Aggregate true labels and probabilities for later threshold application
            aggregated_true_labels.extend(y[test])
            aggregated_probabilities.extend(y_proba)

        # Convert lists to numpy arrays for easier handling
        aggregated_true_labels = np.array(aggregated_true_labels)
        aggregated_probabilities = np.array(aggregated_probabilities)

        # Assume self.average_threshold is already calculated
        avg_threshold = self.average_threshold

        # Convert probabilities to binary predictions using both the default and custom average thresholds
        default_predictions = (aggregated_probabilities >= 0.5).astype(int)
        custom_predictions = (aggregated_probabilities >= avg_threshold).astype(int)

        # Compute confusion matrices for both default and custom threshold predictions
        cm_default = confusion_matrix(aggregated_true_labels, default_predictions)
        cm_custom = confusion_matrix(aggregated_true_labels, custom_predictions)

        # For the default threshold
        TP_default = cm_default[1, 1]
        FN_default = cm_default[1, 0]
        FP_default = cm_default[0, 1]
        TN_default = cm_default[0, 0]

        sensitivity_default = TP_default / (TP_default + FN_default)
        specificity_default = TN_default / (TN_default + FP_default)

        # For the custom threshold
        TP_custom = cm_custom[1, 1]
        FN_custom = cm_custom[1, 0]
        FP_custom = cm_custom[0, 1]
        TN_custom = cm_custom[0, 0]

        sensitivity_custom = TP_custom / (TP_custom + FN_custom)
        specificity_custom = TN_custom / (TN_custom + FP_custom)


        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Plotting the confusion matrix for the default threshold
        sns.heatmap(cm_default, annot=True, fmt="d", cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted labels')
        axes[0].set_ylabel('True labels')
        axes[0].set_title(f'Confusion Matrix {self.model_name} (Default Threshold 0.5)\nSensitivity: {sensitivity_default:.2f}, Specificity: {specificity_default:.2f}', fontsize=10)
        axes[0].set_xticklabels(['Negative', 'Positive'], fontsize=10)
        axes[0].set_yticklabels(['Negative', 'Positive'], rotation=0, fontsize=10)

        # Plotting the confusion matrix for the custom average threshold
        sns.heatmap(cm_custom, annot=True, fmt="d", cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted labels')
        axes[1].set_title(f'({self.threshold_metric} Threshold {avg_threshold:.2f})\nSensitivity: {sensitivity_custom:.2f}, Specificity: {specificity_custom:.2f}', fontsize=10)

        axes[1].set_xticklabels(['Negative', 'Positive'], fontsize=10)
        axes[1].set_yticklabels(['Negative', 'Positive'], rotation=0, fontsize=10)

        plt.tight_layout()
        plt.show()

    def print_scores(self):
        print(f"\nMetrics for the {self.model_name} model (before threshold):")
        for metric in self.pre_cv_scores:
            print(f"{metric.capitalize()}: {self.pre_cv_scores[metric]:.4f} (std: {self.pre_cv_std[metric]:.4f})")

        print(f"\nMetrics for the {self.model_name} model (after threshold):")
        for metric in self.post_cv_scores:
            print(f"{metric.capitalize()}: {self.post_cv_scores[metric]:.4f} (std: {self.post_cv_std[metric]:.4f})")
