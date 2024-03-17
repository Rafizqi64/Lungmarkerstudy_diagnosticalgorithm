import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, auc, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)

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

    def train_voting_classifier(self):
        self.reset()  # Reset the model state before training
        estimators = [(name, model) for name, model in self.trained_models.items()]
        self.voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []

        # Initialize an empty list to store predictions and probabilities for all folds
        all_predictions = []
        all_probabilities = []

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.voting_classifier.fit(X_train, y_train)
            y_pred = self.voting_classifier.predict(X_test)
            y_proba = self.voting_classifier.predict_proba(X_test)[:, 1]

            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))

            # Store predictions and probabilities for this fold
            all_predictions.append((y_test, y_pred))
            all_probabilities.append((y_test, y_proba))

        # After cross-validation, store the predictions and probabilities in self.results
        self.results['predictions'] = all_predictions
        self.results['probabilities'] = all_probabilities

        # Calculating the mean of the metrics
        metrics = {
            "Accuracy": np.mean(accuracies),
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "F1": np.mean(f1_scores),
            "ROC AUC": np.mean(roc_aucs),
        }
        print(f'Metrics for {self.model_name} Model')
        print(metrics)

    def plot_roc_curves(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 8))

        for i, (train, test) in enumerate(cv.split(self.X, self.y)):
            self.voting_classifier.fit(self.X.iloc[train], self.y.iloc[train])
            y_proba = self.voting_classifier.predict_proba(self.X.iloc[test])
            fpr, tpr, _ = roc_curve(self.y.iloc[test], y_proba[:, 1])
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
        plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve {self.model_name} model')
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
        if not self.results:
            print("No results found for the voting model")
            return

        # Assuming 'probabilities' contains the probabilities for the positive class across all CV folds
        probabilities = np.concatenate([proba for _, proba in self.results['probabilities']])
        true_labels = np.concatenate([y_test for y_test, _ in self.results['probabilities']])

        plt.figure(figsize=(15, 7))
        sns.histplot(probabilities[true_labels == 0], bins=20, kde=True, label='Negatives', color='blue', alpha=0.5)
        sns.histplot(probabilities[true_labels == 1], bins=20, kde=True, label='Positives', color='red', alpha=0.7)

        plt.title('Probability Distribution for the Voting Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 1)
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
        X_feature = X.loc[:, self.score_ensemble_features]  # Notice the change here
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X_feature, y)):
            self.model.fit(X_feature.iloc[train], y[train])  # Use .iloc for integer-location based indexing here
            viz = RocCurveDisplay.from_estimator(
                self.model,
                X_feature.iloc[test],  # Use .iloc for integer-location based indexing here
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

    def print_scores(self):
        best_roc_auc_score = self.cv_scores['roc_auc']
        print(f"\nTraining {self.model_name} model...")
        for metric, score in self.cv_scores.items():
            print(f"{metric.capitalize()}: {score:.3f}")


