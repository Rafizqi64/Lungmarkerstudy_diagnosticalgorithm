import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor


class BrockAndHerderModel:
    def __init__(self, filepath, target, binary_map):
        self.preprocessor = DataPreprocessor(filepath, target, binary_map)
        self.model_brock = LogisticRegression(solver='liblinear', random_state=42)
        self.model_herder = LogisticRegression(solver='liblinear', random_state=42)

    def train_with_cross_validation(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        model = None
        best_score = 0  # Adjust based on which metric you prioritize

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = LogisticRegression(solver='liblinear', random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # Store scores for this fold
            scores['accuracy'].append(accuracy_score(y_test, predictions))
            scores['precision'].append(precision_score(y_test, predictions))
            scores['recall'].append(recall_score(y_test, predictions))
            scores['f1'].append(f1_score(y_test, predictions))
            scores['roc_auc'].append(roc_auc_score(y_test, proba))

        # Calculate average scores across folds
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}

        # Assuming best model selection based on average ROC AUC; adjust as needed
        if avg_scores['roc_auc'] > best_score:
            best_score = avg_scores['roc_auc']
            best_model = model

        print(f"Best Average ROC AUC Score: {best_score}")
        # Optionally print other scores
        for metric, score in avg_scores.items():
            print(f"{metric.capitalize()} (average): {score:.4f}")

        return best_model

    def train_models(self):
        # Load and preprocess the data
        X, y = self.preprocessor.load_and_transform_data()
        brock_features = [
        'Family History of LC', 'Current/Former smoker', 'Emphysema',
        'Nodule size (1-30 mm)', 'Nodule Upper Lobe', 'Nodule Count',
        'Nodule Type']
        herder_features = [
        'Current/Former smoker', 'Previous History of Extra-thoracic Cancer',
        'Nodule size (1-30 mm)', 'Nodule Upper Lobe', 'Spiculation',
        'PET-CT Findings']

        # Obtain list of indices for brock and herder features
        brock_indices = self.preprocessor.get_feature_indices(brock_features)
        herder_indices = self.preprocessor.get_feature_indices(herder_features)

        # Select features for Brock and Herder models
        X_brock = X[:, brock_indices]
        X_herder = X[:, herder_indices]

        self.model_brock = self.train_with_cross_validation(X_brock, y)
        self.model_herder = self.train_with_cross_validation(X_herder, y)

    def get_models(self):
        print('Brock and Herder models finished training')
        return self.model_brock, self.model_herder


