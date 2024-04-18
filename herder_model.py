import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold


class HerderModel(BaseEstimator, ClassifierMixin):
    """
    Implements a two-stage predictive model using logistic regression. The model comprises a main classifier
    (MCP model) and a secondary classifier (Herder model), each with selected features and logistic regression.

    The MCP model is trained first, and its predictions are used as inputs along with other selected features
    for the Herder model. This architecture allows for hierarchical feature utilization and decision making.

    Methods:
    - select_features_rf: Selects features based on importance using a Random Forest classifier.
    - select_features_l1: Selects features using L1 regularization to enforce sparsity.
    - select_features_rfe: Selects features using Recursive Feature Elimination with cross-validation.
    - fit: Trains the MCP and Herder models on the given data.
    - predict: Predicts the class labels for the provided data.
    - predict_proba: Predicts class probabilities for the provided data.
    - get_all_selected_features: Returns a combined list of all selected features for both models.
    - print_model_formulae: Prints the logistic regression formulae for both models.

    Attributes:
    - preprocessor: The preprocessing object to apply to data before training models.
    - n_splits: Number of splits for cross-validation.
    - mcp_model, herder_model: Logistic regression models for MCP and Herder stages.
    - mcp_features, herder_features: Lists of feature names used by the MCP and Herder models.
    - model_metrics: Dictionary to store performance metrics of the model.
    - is_fitted: Boolean flag to indicate if the models have been fitted.
    """
    def __init__(self, preprocessor, n_splits=5):
        self.preprocessor = preprocessor
        self.n_splits = n_splits
        self.mcp_model = LogisticRegression(solver='liblinear')
        self.herder_model = LogisticRegression(solver='liblinear')
        self.is_fitted = False
        self.mcp_features = [
            'remainder__Current/Former smoker',
            'remainder__Previous History of Extra-thoracic Cancer',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Spiculation',
            ]

        self.herder_features = [
            'cat__PET-CT Findings_Faint',
            'cat__PET-CT Findings_Intense',
            'cat__PET-CT Findings_Moderate',
            ]
        self.model_metrics = {}


    def select_features_rf(self, X, y):
        """
        Applies feature selection using a Random Forest classifier.
        """
        print("Applying Random Forest for feature selection...")

        # Assuming you have defined mcp_features and herder_features
        for feature_set_name in ['mcp_features', 'herder_features']:
            X_selected = X[getattr(self, feature_set_name)]
            # Initialize Random Forest to estimate feature importances
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_selected, y)

            # Select features based on importance weights
            selector = SelectFromModel(rf, prefit=True)
            selected_features_mask = selector.get_support()
            selected_features = X_selected.columns[selected_features_mask]

            # Update the feature sets based on the selection
            setattr(self, feature_set_name, list(selected_features))
            print(f"Updated {feature_set_name}:", getattr(self, feature_set_name))

    def select_features_l1(self, X, y):
        """
        Applies L1 regularization (via LogisticRegressionCV) for feature selection.
        """
        print("Applying L1 regularization for feature selection...")

        # Assuming you have defined mcp_features and herder_features
        for feature_set_name in ['mcp_features', 'herder_features']:
            X_selected = X[getattr(self, feature_set_name)]
            # Initialize Logistic Regression with L1 penalty using cross-validation
            l1_model = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=42)
            l1_model.fit(X_selected, y)

            # Identify non-zero coefficients (selected features)
            selected_features_mask = l1_model.coef_.flatten() != 0
            selected_features = X_selected.columns[selected_features_mask]

            # Update the feature sets based on the selection
            setattr(self, feature_set_name, list(selected_features))
            print(f"Updated {feature_set_name}:", getattr(self, feature_set_name))

    def select_features_rfe(self, X, y):
        """
        Applies RFECV for feature selection on both MCP and Herder feature sets.
        Updates the feature sets based on the selection results.
        """

        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Apply RFECV to MCP features
        mcp_rfecv = RFECV(estimator=LogisticRegression(solver='liblinear', random_state=42),
                          step=1,
                          cv=cv_strategy,
                          scoring='roc_auc')
        mcp_X = X[self.mcp_features]
        mcp_rfecv.fit(mcp_X, y)
        # Update mcp_features based on the selection
        self.mcp_features = list(mcp_X.columns[mcp_rfecv.support_])

        print("Updated MCP features:", self.mcp_features)

    def get_all_selected_features(self):
        # Combine MCP and Herder features
        return self.mcp_features + self.herder_features

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the MCP model
            self.mcp_model.fit(X_train[self.mcp_features], y_train)
            # Get MCP predictions for the Herder model input
            mcp_predictions_train = self.mcp_model.predict_proba(X_train[self.mcp_features])[:, 1]

            # Prepare Herder model training data
            X_train_herder = X_train[self.herder_features].copy()
            X_train_herder['MCP_Output'] = mcp_predictions_train
            self.herder_model.fit(X_train_herder, y_train)

            # Collect metrics for the current fold
            fold_metric = self._evaluate_fold(X_train_herder, X_test, y_train, y_test)
            fold_metrics.append(fold_metric)

        # Calculate and store aggregated metrics across all folds
        self.calculate_and_store_metrics("HerderModel", fold_metrics)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("The model is not fitted yet.")


        mcp_predictions = self.mcp_model.predict_proba(X[self.mcp_features])[:, 1]

        X_herder = X[self.herder_features].copy()
        X_herder['MCP_Output'] = mcp_predictions

        return self.herder_model.predict_proba(X_herder)

    def predict(self, X):
        # Check if the model is fitted
        if not self.is_fitted:
            raise RuntimeError("You must train classifier before predicting data!")

        # Generate predictions using the logic based on predict_proba
        probas = self.predict_proba(X)
        # Return the class with the highest probability
        return np.argmax(probas, axis=1)

    def _evaluate_fold(self, X_train, X_test, y_train, y_test):
        # Predictions for training data
        y_train_pred = self.herder_model.predict(X_train)
        y_train_proba = self.herder_model.predict_proba(X_train)[:, 1]
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)

        # Predictions for test data
        mcp_predictions_test = self.mcp_model.predict_proba(X_test[self.mcp_features])[:, 1]
        X_test_herder = X_test[self.herder_features].copy()
        X_test_herder['MCP_Output'] = mcp_predictions_test
        y_test_pred = self.herder_model.predict(X_test_herder)
        y_test_proba = self.herder_model.predict_proba(X_test_herder)[:, 1]
        val_fpr, val_tpr, _ = roc_curve(y_test, y_test_proba)
        val_roc_auc = roc_auc_score(y_test, y_test_proba)

        return (y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba,
                train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc)

    def calculate_and_store_metrics(self, model_name, fold_metrics):
        # Initialization
        accuracies, precisions, recalls, f1_scores, train_roc_aucs, val_roc_aucs = [], [], [], [], [], []

        # ROC data initialization
        roc_data = {'train': {'fpr': [], 'tpr': [], 'roc_auc': []},
                    'validation': {'fpr': [], 'tpr': [], 'roc_auc': []}}

        for metrics in fold_metrics:
            y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba, \
            train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc = metrics

            accuracies.append(accuracy_score(y_test, y_test_pred))
            precisions.append(precision_score(y_test, y_test_pred))
            recalls.append(recall_score(y_test, y_test_pred))
            f1_scores.append(f1_score(y_test, y_test_pred))
            train_roc_aucs.append(train_roc_auc)
            val_roc_aucs.append(val_roc_auc)

            roc_data['train']['fpr'].append(train_fpr)
            roc_data['train']['tpr'].append(train_tpr)
            roc_data['train']['roc_auc'].append(train_roc_auc)
            roc_data['validation']['fpr'].append(val_fpr)
            roc_data['validation']['tpr'].append(val_tpr)
            roc_data['validation']['roc_auc'].append(val_roc_auc)

        mean_metrics = {
            "Accuracy": np.mean(accuracies),
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "F1": np.mean(f1_scores),
            "Train ROC AUC": np.mean(train_roc_aucs),
            "Validation ROC AUC": np.mean(val_roc_aucs),
        }
        std_metrics = {
            "Accuracy STD": np.std(accuracies),
            "Precision STD": np.std(precisions),
            "Recall STD": np.std(recalls),
            "F1 STD": np.std(f1_scores),
            "Train ROC AUC STD": np.std(train_roc_aucs),
            "Validation ROC AUC STD": np.std(val_roc_aucs),
        }

        # Print and store metrics including standard deviations
        print(f"Metrics for {model_name} Model:")
        for metric, value in mean_metrics.items():
            std_metric = std_metrics.get(metric + " STD", 0)  # Default to 0 if not found
            print(f"{metric}: {value:.4f} (std: {std_metric:.4f})")

        self.model_metrics[model_name] = {
            'metrics': mean_metrics,
            'metrics_std': std_metrics,
            'roc_data': roc_data,
        }

    def print_model_formulae(self):
        if not self.is_fitted:
            raise RuntimeError("The model is not fitted yet. Please fit the model before obtaining the formulae.")

        # Function to format the logistic regression formula as a string
        def format_formula(feature_names, coefficients, intercept):
            terms = [f"{coeff:.4f}*{name}" for name, coeff in zip(feature_names, coefficients)]
            formula = " + ".join(terms)
            return f"logit(P(y=1)) = {intercept:.4f} + {formula}"

        # MCP Model Formula
        mcp_formula = format_formula(self.mcp_features, self.mcp_model.coef_[0], self.mcp_model.intercept_[0])
        print("MCP Model Formula:")
        print(mcp_formula)

        # Herder Model Formula
        # Note: We add 'MCP_Output' to the herder_features for formula generation
        herder_features_with_mcp = self.herder_features + ['MCP_Output']
        herder_formula = format_formula(herder_features_with_mcp, self.herder_model.coef_[0], self.herder_model.intercept_[0])
        print("\nHerder Model Formula:")
        print(herder_formula)
