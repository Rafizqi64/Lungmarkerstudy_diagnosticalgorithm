import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, auc,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_preprocessing import DataPreprocessor


class VotingModel:
    def __init__(self, trained_models, ensemble_features, filepath, target, binary_map, model_name):
        self.data_preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.trained_models = trained_models
        self.ensemble_features = ensemble_features
        self.voting_classifier = None
        self.results = {}
        self.X = None
        self.y = None
        self.load_data()
        self.model_name = model_name

    def reset(self):
        self.voting_classifier = None
        self.results = {}

    def load_data(self):
        self.X, self.y = self.data_preprocessor.load_and_transform_data()
        self.X = self.X[self.ensemble_features]

#     def train_voting_classifier(self, threshold_metric='ppv', desired_percentage=0.95):
        # self.reset()
        # estimators = [(name, model) for name, model in self.trained_models.items()]
        # self.voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # metrics_before, metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}, {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
        # thresholds = []
        # aggregated_probabilities = []
        # aggregated_y_test = []

        # for train_index, test_index in skf.split(self.X, self.y):
            # X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            # y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            # self.voting_classifier.fit(X_train, y_train)

            # y_train_proba = self.voting_classifier.predict_proba(X_train)[:, 1]
            # y_proba = self.voting_classifier.predict_proba(X_test)[:, 1]
            # y_pred = (y_proba >= 0.5).astype(int)
            # aggregated_probabilities.extend(y_proba.tolist())
            # aggregated_y_test.extend(y_test.tolist())

            # metrics_before['accuracy'].append(accuracy_score(y_test, y_pred))
            # metrics_before['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            # metrics_before['recall'].append(recall_score(y_test, y_pred))
            # metrics_before['f1'].append(f1_score(y_test, y_pred))
            # metrics_before['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
            # metrics_before['val_roc_auc'].append(roc_auc_score(y_test, y_proba))

            # # Determine and apply custom threshold based on the specified metric
            # if threshold_metric == 'ppv':
                # precision, recall, thresholds_pr = precision_recall_curve(y_train, y_train_proba)
                # threshold_indices = np.where(precision >= desired_percentage)[0]
                # threshold = thresholds_pr[threshold_indices[0] - 1] if threshold_indices.size > 0 else 1.0
            # elif threshold_metric == 'npv':
                # # Adjusted NPV calculation logic
                # thresholds_pr = np.linspace(0, 1, 100)
                # best_threshold = 0.5  # Default threshold if no better option is found
                # best_npv = 0  # Track the best NPV found

                # for threshold in thresholds_pr:
                    # # Adjust predictions based on the threshold
                    # y_pred_train_adjusted = (y_train_proba >= threshold).astype(int)
                    # tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train_adjusted).ravel()

                    # npv_adjusted = tn / (tn + fn) if (tn + fn) > 0 else 0

                    # # Update the best threshold if this NPV is better and meets/exceeds the desired percentage
                    # if npv_adjusted > best_npv and npv_adjusted >= desired_percentage:
                        # best_threshold = threshold
                        # best_npv = npv_adjusted
                # # Use the best threshold found
                # threshold = best_threshold
            # else:
                # raise ValueError("Invalid threshold_metric. Choose 'ppv' or 'npv'.")
            # thresholds.append(threshold)

            # y_pred_adjusted = (y_proba >= threshold).astype(int)
            # metrics_after['accuracy'].append(accuracy_score(y_test, y_pred_adjusted))
            # metrics_after['precision'].append(precision_score(y_test, y_pred_adjusted, zero_division=0))
            # metrics_after['recall'].append(recall_score(y_test, y_pred_adjusted))
            # metrics_after['f1'].append(f1_score(y_test, y_pred_adjusted))
            # metrics_after['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
            # metrics_after['val_roc_auc'].append(roc_auc_score(y_test, y_proba))

        # self.results['probabilities'] = np.array(aggregated_probabilities)
        # self.results['y_test'] = np.array(aggregated_y_test)
        # self.results['thresholds'] = thresholds
        # print(self.results['thresholds'])
        # self.results['average_threshold'] = np.mean(thresholds)

        # self.print_metrics(f"(before {threshold_metric} threshold)", metrics_before)
        # self.print_metrics(f"(after {threshold_metric} threshold)", metrics_after)
        # avg_threshold = np.mean(thresholds)
        # print(f"\naverage {threshold_metric} threshold used for the {self.model_name} model: {avg_threshold:.4f}")

    def train_voting_classifier(self, threshold_metric='ppv', desired_percentage=0.95):
        self.reset()
        estimators = [(name, model) for name, model in self.trained_models.items()]
        self.voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metrics_before, metrics_after = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}, {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_roc_auc': [], 'val_roc_auc': []}
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

            metrics_before['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_before['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics_before['recall'].append(recall_score(y_test, y_pred))
            metrics_before['f1'].append(f1_score(y_test, y_pred))
            metrics_before['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
            metrics_before['val_roc_auc'].append(roc_auc_score(y_test, y_proba))

            # Determine and apply custom threshold based on the specified metric
            if threshold_metric == 'ppv':
                precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
                threshold_indices = np.where(precision >= desired_percentage)[0]
                threshold = thresholds_pr[threshold_indices[0] - 1] if threshold_indices.size > 0 else 1.0
            elif threshold_metric == 'npv':
                # Calculate NPV and adjust threshold
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                thresholds_pr = sorted(list(set(y_proba)))
                for threshold in thresholds_pr[::-1]:
                    y_pred_adjusted = (y_proba >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adjusted).ravel()
                    npv_adjusted = tn / (tn + fn) if (tn + fn) > 0 else 0
                    if npv_adjusted >= desired_percentage:
                        break
            else:
                raise ValueError("Invalid threshold_metric. Choose 'precision' or 'npv'.")
            thresholds.append(threshold)

            y_pred_adjusted = (y_proba >= threshold).astype(int)
            metrics_after['accuracy'].append(accuracy_score(y_test, y_pred_adjusted))
            metrics_after['precision'].append(precision_score(y_test, y_pred_adjusted, zero_division=0))
            metrics_after['recall'].append(recall_score(y_test, y_pred_adjusted))
            metrics_after['f1'].append(f1_score(y_test, y_pred_adjusted))
            metrics_after['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
            metrics_after['val_roc_auc'].append(roc_auc_score(y_test, y_proba))

        self.results['probabilities'] = np.array(aggregated_probabilities)
        self.results['y_test'] = np.array(aggregated_y_test)
        self.results['thresholds'] = thresholds
        self.results['average_threshold'] = np.mean(thresholds)

        self.print_metrics(f"(before {threshold_metric} threshold)", metrics_before)
        self.print_metrics(f"(after {threshold_metric} threshold)", metrics_after)
        avg_threshold = np.mean(thresholds)
        print(f"\naverage ppv threshold used for the {self.model_name} model: {avg_threshold:.4f}")

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
        plt.scatter(avg_threshold, interp_mean_tpr_at_threshold, color='black', label=f'Avg. Threshold at FPR={avg_threshold:.2f}, TPR={interp_mean_tpr_at_threshold:.2f}')

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
        # Ensure your voting classifier is already trained and your data (self.X) is prepared
        if not self.voting_classifier:
            print("Voting classifier is not trained.")
            return

        # SHAP Explainer initialization may vary based on the model type
        explainer = shap.Explainer(self.voting_classifier.predict, self.X)
        shap_values = explainer(self.X)

        # Plotting
        shap.plots.bar(shap_values, max_display=30)
        shap.plots.beeswarm(shap_values, max_display=30, show=False)
        plt.title("Ensemble model SHAP Beeswarm Plot", fontsize=20)
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
            plt.axvline(x=average_threshold, color='green', linestyle='--', label=f'Threshold: {average_threshold:.2f}')

        plt.title('Probability Distribution for the Voting Model', fontsize=20)
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

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plotting the confusion matrix for the default threshold
        sns.heatmap(cm_default, annot=True, fmt="d", ax=axes[0], cmap=plt.cm.Blues, cbar=False)
        axes[0].set_title(f'Confusion Matrix {self.model_name} (Default Threshold 0.5)', fontsize=16)
        axes[0].set_xlabel('Predicted Label', fontsize=14)
        axes[0].set_ylabel('True Label', fontsize=14)
        axes[0].xaxis.set_ticklabels(['Negative', 'Positive'])
        axes[0].yaxis.set_ticklabels(['Negative', 'Positive'])

        # Plotting the confusion matrix for the custom average threshold
        sns.heatmap(cm_custom, annot=True, fmt="d", ax=axes[1], cmap=plt.cm.Blues, cbar=False)
        axes[1].set_title(f'Confusion Matrix (Custom Threshold {average_threshold:.2f})', fontsize=16)
        axes[1].set_xlabel('Predicted Label', fontsize=14)
        axes[1].set_ylabel('True Label', fontsize=14)
        axes[1].xaxis.set_ticklabels(['Negative', 'Positive'])
        axes[1].yaxis.set_ticklabels(['Negative', 'Positive'])

        plt.tight_layout()
        plt.show()

class score_based_ensemble:
    def __init__(self, filepath, target, binary_map, features, model_name):
        self.model = LogisticRegression(solver='liblinear')
        self.cv_scores = {}
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.score_ensemble_features = features
        self.model_name = model_name

    def fit_evaluate(self, n_splits=5, random_state=42, scoring=None):

        X, y = self.preprocessor.load_and_transform_data()

        # Select features for Brock and Herder models
        X_feature = X[self.score_ensemble_features]
        if scoring is None:
            scoring = {'accuracy': 'accuracy',
                       'precision': 'precision',
                       'recall': 'recall',
                       'f1' : 'f1',
                       'roc_auc': 'roc_auc'}
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = cross_validate(self.model, X_feature, y, cv=cv, scoring=scoring, return_train_score=False)
        for metric, score_name in scoring.items():
            metric_key = 'test_' + score_name
            self.cv_scores[metric] = np.mean(cv_results[metric_key])

    def plot_roc_curve(self, n_splits=5, random_state=42):
        X, y = self.preprocessor.load_and_transform_data()

        # Correctly select features using .loc for label-based indexing
        X_feature = X.loc[:, self.score_ensemble_features]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X_feature, y)):
            self.model.fit(X_feature.iloc[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                self.model,
                X_feature.iloc[test],
                y[test],
                name=f"ROC fold {fold}",
                alpha=0.6,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.3,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curve across CV folds for {self.model_name} model",
        )
        ax.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.show()

    def plot_prediction_histogram(self):
        """
        Plot histogram of the predicted probabilities for the ensemble model.
        """
        X, y = self.preprocessor.load_and_transform_data()

        # Select features for Brock and Herder models
        X_ensemble = X.loc[:, self.score_ensemble_features]
        # Fit the model on the provided data
        self.model.fit(X_ensemble, y)

        # Predict probabilities
        probabilities = self.model.predict_proba(X_ensemble)[:, 1]

        # Set the visual theme for the seaborn plots
        sns.set(style="whitegrid")

        plt.figure(figsize=(15, 7))
        # Use probabilities and actual labels (y) to plot the histograms
        sns.histplot(probabilities[y == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(probabilities[y == 1], bins=20, kde=True, label='Positives', alpha=0.7, color='red')
        plt.title(f'Probability Distribution for {self.model_name} Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlim(0, 1)  # Ensure the x-axis ranges from 0 to 1
        plt.show()

    def plot_confusion_matrix(self, n_splits=5, random_state=42):
        X, y = self.preprocessor.load_and_transform_data()

        # Select features for the ensemble model
        X_feature = X.loc[:, self.score_ensemble_features]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        conf_matrix_sum = np.zeros((2, 2))  # Initialize the confusion matrix sum

        for train, test in cv.split(X_feature, y):
            self.model.fit(X_feature.iloc[train], y[train])
            y_pred = self.model.predict(X_feature.iloc[test])
            conf_matrix_sum += confusion_matrix(y[test], y_pred)

        # Compute the mean confusion matrix over the folds
        conf_matrix = conf_matrix_sum / n_splits

        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='0.2f', cmap='Blues', ax=ax)

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Average Confusion Matrix for {self.model_name} over {n_splits} CV folds')
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.yaxis.set_ticklabels(['Negative', 'Positive'])

        plt.show()


    def print_scores(self):
        best_roc_auc_score = self.cv_scores['roc_auc']
        print(f"\nTraining {self.model_name} model...")
        for metric, score in self.cv_scores.items():
            print(f"{metric.capitalize()}: {score:.3f}")


