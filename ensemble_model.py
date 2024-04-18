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
    """
    Manages the training and evaluation of an ensemble voting classifier using features and models specified from the model class. This class
    integrates multiple trained models to predict outcomes based on a majority voting system. The class facilitates
    evaluating the ensemble model with performance metrics and visualizing ROC curves and prediction distributions.

    Attributes:
    - trained_models (dict): A dictionary of pre-trained model objects.
    - ensemble_features (list): A list of features used by the ensemble for making predictions.
    - filepath (str): Path to the dataset file.
    - target (str): The Diagnosis variable.
    - binary_map (dict): A dictionary mapping the diagnosis' classes to binary labels.
    - model_name (str): Name assigned to the ensemble model for identification.
    - threshold_metric (str, optional): Specifies the metric to adjust the decision threshold ('ppv' or 'npv'). Default is 'ppv'.

    Methods:
    - reset: Resets the voting classifier and results.
    - load_data: Loads and preprocesses data from a specified filepath.
    - train_voting_classifier: Trains a soft voting classifier using the specified models and evaluates performance across standard and custom thresholds.
    - determine_custom_threshold: Calculates a threshold that meets a specified minimum metric percentage.
    - append_metrics: Appends performance metrics after model evaluation.
    - print_metrics: Prints collected metrics in a formatted manner.
    - plot_roc_curves: Visualizes the Receiver Operating Characteristic (ROC) curves for the voting classifier.
    - generate_shap_plot: Generates SHAP plots to explain the impact of features on the predictions.
    - plot_prediction_histograms: Visualizes the distribution of predicted probabilities.
    - plot_confusion_matrices: Displays confusion matrices for the default and custom threshold predictions to evaluate model performance visually.
    """
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
        """
        Trains a soft voting classifier and evaluates it using Stratified K-Fold cross-validation.
        Calculates performance metrics at default (0.5) and custom thresholds to achieve a specified
        minimum percentage of positive predictions.

        Parameters:
        - desired_percentage (float, optional): Target percentage of positive class predictions, default is 0.95.

        Modifies:
        - self.voting_classifier: Initialized with models from `self.trained_models`.
        - self.results: Stores probabilities, test labels, and thresholds.

        Outputs:
        - Prints performance metrics before and after applying the custom threshold.

        Returns:
        - None
        """
        self.reset()
        estimators = [(name, model) for name, model in self.trained_models.items()]
        self.voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metrics_before = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
        metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
        thresholds = []
        aggregated_probabilities = []
        aggregated_y_test = []

        # Collect metrics and predictions for each fold
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.voting_classifier.fit(X_train, y_train)

            y_train_proba = self.voting_classifier.predict_proba(X_train)[:, 1]
            y_proba = self.voting_classifier.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            aggregated_probabilities.extend(y_proba.tolist())
            aggregated_y_test.extend(y_test.tolist())

            self.append_metrics(metrics_before, y_train, y_train_proba, y_test, y_proba, y_pred)
            threshold = self.determine_custom_threshold(y_test, y_proba, desired_percentage)
            thresholds.append(threshold)

            y_pred_adjusted = (y_proba >= threshold).astype(int)
            self.append_metrics(metrics_after, y_train, y_train_proba, y_test, y_proba, y_pred_adjusted)

        # Finalizing results and printing metrics
        self.results['probabilities'] = np.array(aggregated_probabilities)
        self.results['y_test'] = np.array(aggregated_y_test)
        self.results['thresholds'] = thresholds
        self.results['average_threshold'] = np.mean(thresholds)
        self.print_metrics("(before custom threshold)", metrics_before)
        self.print_metrics("(after custom threshold)", metrics_after)

    def determine_custom_threshold(self, y_test, y_proba, desired_percentage):
        """
        Determines the threshold for predictions that achieves a specified minimum performance metric percentage.

        This function selects the optimal threshold based on the performance metric specified in `self.threshold_metric`.
        For 'ppv' (positive predictive value), it uses precision-recall curve for each fold. For 'npv' (negative predictive value),
        it calculates the NPV for each threshold and selects the one with the best performance DOES NOT INCLUDE AVERAGING YET.

        Parameters:
        - y_test (array-like): True binary labels.
        - y_proba (array-like): Predicted probabilities for the positive class.
        - desired_percentage (float): The desired minimum percentage for the chosen performance metric.

        Returns:
        - float: The threshold that meets the desired performance metric percentage, defaulting to 1.0 for PPV if no
          valid threshold is found, and to 0.5 for NPV.
        """
        if self.threshold_metric == 'ppv':
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
            threshold_indices = np.where(precision >= desired_percentage)[0]
            return thresholds_pr[threshold_indices[0] - 1] if threshold_indices.size > 0 else 1.0
        elif self.threshold_metric == 'npv':
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
            sorted_indices = np.argsort(y_proba)
            sorted_proba = y_proba[sorted_indices]
            sorted_y_test_np = y_test_np[sorted_indices]

            best_threshold = None
            best_npv = 0

            for idx, threshold in enumerate(sorted_proba[:-1]):
                y_pred = (sorted_proba > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(sorted_y_test_np, y_pred).ravel()
                if (tn + fn) > 0:
                    npv = tn / (tn + fn)
                    if npv >= desired_percentage and npv > best_npv:
                        best_npv = npv
                        best_threshold = threshold

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

        plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

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
            plt.axvline(x=0.5, color='green', linestyle='--', label=f'Threshold: 0.5')

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

        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm_default, annot=True, fmt="d", cmap='Blues')
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title(f'Confusion Matrix {self.model_name} (Threshold 0.5)\nSensitivity: {sensitivity_default:.2f}, Specificity: {specificity_default:.2f}', fontsize=10)
        # plt.xticks(ticks=np.arange(2) + 0.5, labels=['Negative', 'Positive'], fontsize=10)
        # plt.yticks(ticks=np.arange(2) + 0.5, labels=['Negative', 'Positive'], rotation=0, fontsize=10)


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
