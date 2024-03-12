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
                                     cross_validate, learning_curve)

from data_preprocessing import DataPreprocessor


class ensemble_model:
    def __init__(self, models, filepath, target, binary_map, voting='soft'):
        self.ensemble = VotingClassifier(estimators=models, voting=voting)
        self.best_model = None
        self.cv_results = {}
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.ensemble_features = [
           'cat__Nodule Type_GroundGlass', 'cat__Nodule Type_PartSolid',
           'cat__Nodule Type_Solid', 'cat__PET-CT Findings_Faint',
           'cat__PET-CT Findings_Intense', 'cat__PET-CT Findings_Moderate',
           'cat__PET-CT Findings_No FDG avidity',
           'remainder__Family History of LC', 'remainder__Current/Former smoker',
           'remainder__Previous History of Extra-thoracic Cancer',
           'remainder__Emphysema', 'remainder__Nodule size (1-30 mm)',
           'remainder__Nodule Upper Lobe', 'remainder__Nodule Count',
           'remainder__Spiculation',
           'remainder__CEA', 'remainder__CYFRA 21-1',
           'remainder__NSE',
           'remainder__proGRP'
            ]

    def fit_best_model(self, n_splits=5, random_state=42, scoring='roc_auc'):
        X, y = self.preprocessor.load_and_transform_data()  # Assume this method is defined in DataPreprocessor
        X_ensemble = X[self.ensemble_features]

        # Initialize container for cross-validation results including model estimators
        self.model_results = []

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_index, test_index in cv.split(X_ensemble, y):
            X_train, X_test = X_ensemble.iloc[train_index], X_ensemble.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.ensemble.fit(X_train, y_train)
            y_proba = self.ensemble.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

            # Store true labels and probabilities for ROC plotting
            self.model_results.append((y_test, y_proba))

        # Perform cross-validation and get average
#         best_estimator = self.tune_hyperparameters(X_ensemble, y)
#         self.ensemble = best_estimator

        # Use cross_validate for generating scores for multiple metrics
        cv_results = cross_validate(self.ensemble, X_ensemble, y, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], return_estimator=True)
# Calculate average scores from cv_results
        avg_scores = {metric: np.mean(scores) for metric, scores in cv_results.items() if 'test_' in metric}
        best_roc_auc = avg_scores['test_roc_auc']

        # Print the formatted results
        print(f"Best ROC AUC Score for Ensemble Model: {best_roc_auc:.4f}")
        for metric, score in avg_scores.items():
            metric_name = metric.replace('test_', '').capitalize()
            print(f"{metric_name} (average) for Ensemble: {score:.4f}")

        best_estimator_index = np.argmax(cv_results['test_roc_auc'])
        best_model = cv_results['estimator'][best_estimator_index]
        self.generate_shap_plot(X_ensemble, best_model)


    def tune_hyperparameters(self, X, y):
        param_grid = {
            'brock__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'herder__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'nsclc__C': [0.001, 0.01, 0.1, 1, 10, 100],
        }
        grid_search = GridSearchCV(self.ensemble, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc', verbose=1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def generate_shap_plot(self, X, model):
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values, max_display=30)

    def plot_learning_curve(self, X, y, title='Learning Curve'):
        train_sizes, train_scores, test_scores = learning_curve(
            self.ensemble, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def plot_roc_curves(self, model_results, model_name='Ensemble Model'):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        fig, ax = plt.subplots()
        for i, (y_test, proba) in enumerate(model_results):
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='Fold %d ROC (AUC = %0.2f)' % (i, roc_auc))

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="ROC Curve for {}".format(model_name))
        ax.legend(loc="lower right")
        plt.show()

    def plot_prediction_histograms(self, predictions, true_labels, title='Probability Distribution', positive_label='Positives', negative_label='Negatives'):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        positive_predictions = predictions[true_labels == 1]
        negative_predictions = predictions[true_labels == 0]

        sns.histplot(negative_predictions, bins=20, kde=True, label=negative_label, color='blue', alpha=0.5)
        sns.histplot(positive_predictions, bins=20, kde=True, label=positive_label, color='red', alpha=0.7)

        plt.title(title)
        plt.xlabel('Probability of being Positive Class')
        plt.ylabel('Density')
        plt.legend()
        plt.show()


class score_based_ensemble:
    def __init__(self, filepath, target, binary_map):
        self.model = LogisticRegression(solver='liblinear')
        self.cv_scores = {}
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.score_ensemble_features = ['remainder__Brock score (%)', 'remainder__Herder score (%)', 'remainder__% LC in TM-model', 'remainder__% NSCLC in TM-model']

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
        feature_indices = self.preprocessor.get_feature_indices(self.score_ensemble_features)

        # Select features for Brock and Herder models
        X_feature = X[:, feature_indices]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X_feature, y)):
            self.model.fit(X_feature[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                self.model,
                X_feature[test],
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
            title="ROC Curve across CV folds for Ensemble model",
        )
        ax.legend(loc="lower right")
        plt.show()

    def plot_prediction_histogram(self):
        """
        Plot histogram of the predicted probabilities for the ensemble model.
        """
        X, y = self.preprocessor.load_and_transform_data()
        ensemble_indices = self.preprocessor.get_feature_indices(self.score_ensemble_features)

        # Select features for Brock and Herder models
        X_ensemble = X[:, ensemble_indices]
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
        plt.title('Probability Distribution for Score Based Ensemble Model', fontsize=20)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlim(0, 1)  # Ensure the x-axis ranges from 0 to 1
        plt.show()

    def print_scores(self):
        best_roc_auc_score = self.cv_scores['roc_auc']
        print(f"Best Average ROC AUC Score: {best_roc_auc_score:.3f}")
        for metric, score in self.cv_scores.items():
            print(f"{metric.capitalize()}: {score:.3f}")


