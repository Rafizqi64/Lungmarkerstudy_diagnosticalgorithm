import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class LBxModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_lc = None
        self.X_nsclc = None
        self.y = None
        # Ensure that the models are properly initialized here
        self.model_lc = LogisticRegression()
        self.model_nsclc = LogisticRegression()
        self.scores_lc = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
        self.scores_nsclc = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}


    #==============#
    # PREPARE DATA #
    #==============#

    def load_and_prepare_data(self):
        # Load dataset
        self.df = pd.read_excel(self.filepath)

        # Apply 10log transformation to protein markers
        for column in ['CYFRA 21-1', 'CEA', 'NSE', 'proGRP']:
            self.df[column] = np.log10(self.df[column])

        self.df['Target'] = self.df['Diagnose'].apply(lambda x: 1 if x == 'NSCLC' else 0)
        self.y = self.df['Target']

        # Define features for both models
        self.X_lc = self.df[['CYFRA 21-1', 'CEA']]
        self.X_nsclc = self.df[['CEA', 'CYFRA 21-1', 'NSE', 'proGRP']]


    #================#
    #    MAIN LOOP   #
    #================#

    def train_and_evaluate(self):
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(self.X_lc, self.y)):
            X_lc_train, X_lc_test = self.X_lc.iloc[train_index], self.X_lc.iloc[test_index]
            y_lc_train, y_lc_test = self.y.iloc[train_index], self.y.iloc[test_index]
            scaler_lc = StandardScaler().fit(X_lc_train)
            X_lc_train_scaled = scaler_lc.transform(X_lc_train)
            X_lc_test_scaled = scaler_lc.transform(X_lc_test)
            self.model_lc.fit(X_lc_train_scaled, y_lc_train)
            predictions_lc = self.model_lc.predict(X_lc_test_scaled)
            proba_lc = self.model_lc.predict_proba(X_lc_test_scaled)[:, 1]
            for score_name in self.scores_lc:
                score_func = globals()[score_name + '_score']
                self.scores_lc[score_name].append(score_func(y_lc_test, predictions_lc, **({'average': 'macro'} if score_name == 'f1' else {})))

            X_nsclc_train, X_nsclc_test = self.X_nsclc.iloc[train_index], self.X_nsclc.iloc[test_index]
            y_nsclc_train, y_nsclc_test = self.y.iloc[train_index], self.y.iloc[test_index]
            scaler_nsclc = StandardScaler().fit(X_nsclc_train)
            X_nsclc_train_scaled = scaler_nsclc.transform(X_nsclc_train)
            X_nsclc_test_scaled = scaler_nsclc.transform(X_nsclc_test)
            self.model_nsclc.fit(X_nsclc_train_scaled, y_nsclc_train)
            predictions_nsclc = self.model_nsclc.predict(X_nsclc_test_scaled)
            proba_nsclc = self.model_nsclc.predict_proba(X_nsclc_test_scaled)[:, 1]
            for score_name in self.scores_nsclc:
                score_func = globals()[score_name + '_score']
                self.scores_nsclc[score_name].append(score_func(y_nsclc_test, predictions_nsclc, **({'average': 'macro'} if score_name == 'f1' else {})))

        # Calculate the average of the scores for each model
        average_scores_lc = {metric: np.mean(scores) for metric, scores in self.scores_lc.items()}
        average_scores_nsclc = {metric: np.mean(scores) for metric, scores in self.scores_nsclc.items()}
        print("Average LC Model Scores:", average_scores_lc)
        print("Average NSCLC Model Scores:", average_scores_nsclc)


    #===============#
    # MODEL FORMULA #
    #===============#

    def get_model_formula(self, model, feature_names):
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]
        formula = f"log(p/(1-p)) = {intercept:.4f} "
        formula += " ".join([f"+ {coef:.4f}*{name}" for coef, name in zip(coefficients, feature_names)])
        return formula

    def print_model_formulas(self):
        # Ensure models are trained
        if self.model_lc is None or self.model_nsclc is None:
            print("Models are not trained yet.")
            return

        lc_formula = self.get_model_formula(self.model_lc, ['CYFRA 21-1', 'CEA'])
        nsclc_formula = self.get_model_formula(self.model_nsclc, ['CEA', 'CYFRA 21-1', 'NSE', 'proGRP'])

        print("LC Model Formula:\n", lc_formula)
        print("\nNSCLC Model Formula:\n", nsclc_formula)


model = LBxModel(filepath='Dataset BEP Rafi.xlsx')
model.load_and_prepare_data()
model.train_and_evaluate()
model.print_model_formulas()
