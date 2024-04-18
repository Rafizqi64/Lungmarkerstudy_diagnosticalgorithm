import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold


class DecisionTree:
    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
        self.categorical_vars = ['Current/Former smoker', 'Emphysema', 'Spiculation', 'PET-CT Findings']
        self.outcome_mapping = {
            'Discharge': 0,
            'CT surveillance': 0,
            'Consider image-guided biopsy': 1,
            'Consider excision or non-surgical treatment': 1,
        }

    def prepare_guideline_data(self):
        """Prepare and return guideline data for comparison."""
        self.apply_guideline_logic()
        self.df['Guideline Scores'] = self.df['Guideline Outcome'].map(self.outcome_mapping)

        if self.df['Guideline Scores'].isnull().any():
            missing_outcomes = self.df[self.df['Guideline Scores'].isnull()]['Guideline Outcome'].unique()
            raise ValueError(f"Missing mappings for outcomes: {missing_outcomes}")

        # Generate a detailed outcomes dataframe with counts for each combination of outcome and diagnosis
        detailed_outcomes = self.df.groupby(['Guideline Outcome', 'Diagnose']).size().unstack(fill_value=0)

        # Calculate total counts for each outcome
        outcome_counts = self.df['Guideline Outcome'].value_counts()
        print(detailed_outcomes)
        # Return the detailed outcomes dataframe and the outcome counts
        return self.evaluate_metrics_cv()

    def evaluate_metrics_cv(self, n_splits=5, random_state=42):
        """Evaluate guideline metrics using cross-validation."""
        y = self.df['Diagnosis_Encoded'].values
        scores = self.df['Guideline Scores'].values  # Use the mapped scores for evaluation

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []  # List to hold metrics for each fold

        for train_index, test_index in cv.split(np.zeros(len(y)), y):
            y_true, y_pred = y[test_index], scores[test_index]

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            fold_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'specificity': specificity
            })

        return fold_metrics

    def apply_guideline_logic(self):
        """Apply the guideline logic to each patient and compare with diagnosis."""
        outcomes = []  # to store the outcome for each patient

        for index, row in self.df.iterrows():
            # Initialize variables for Brock and Herder scores
            brock_score = row['Brock score (%)']
            herder_score = row['Herder score (%)']
            if row['Nodule size (1-30 mm)'] < 5:
                outcome = 'Discharge'
            elif row['Nodule size (1-30 mm)'] < 8:
                outcome = 'CT surveillance'
            elif row['Nodule size (1-30 mm)'] >= 8:
                if brock_score < 10:
                    outcome = 'CT surveillance'
                else:
                    if herder_score < 10:
                        outcome = 'CT surveillance'
                    elif 10 <= herder_score < 70:
                        outcome = 'Consider image-guided biopsy'
                    else:
                        outcome = 'Consider excision or non-surgical treatment'
            else:
                outcome = 'CT surveillance or other as per individual risk and preference'

            outcomes.append(outcome)

        self.df['Guideline Outcome'] = outcomes

        # Generate a detailed outcomes dataframe with counts for each combination of outcome and diagnosis
        detailed_outcomes = self.df.groupby(['Guideline Outcome', 'Diagnose']).size().unstack(fill_value=0)

        # Calculate total counts for each outcome
        outcome_counts = self.df['Guideline Outcome'].value_counts()
        print(detailed_outcomes)
        # Return the detailed outcomes dataframe and the outcome counts
        return self.df, detailed_outcomes, outcome_counts

    def plot_roc_curve(self, n_splits=5, random_state=42):
        self.df['Predicted'] = self.df['Guideline Outcome'].map(self.outcome_mapping)

        y = self.df['Diagnosis_Encoded'].values
        scores = self.df['Predicted'].values

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(10, 8))

        for fold, (train, test) in enumerate(cv.split(np.zeros(shape=(scores.shape[0], 1)), y)):
            fpr, tpr, thresholds = roc_curve(y[test], scores[test])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold + 1} (AUC = {roc_auc:.2f})')

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                         label='± 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve BTS Guidelines')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self):
        self.df['Predicted'] = self.df['Guideline Outcome'].map(self.outcome_mapping)

        y_true = self.df['Diagnosis_Encoded']

        y_pred = self.df['Predicted']

        cm = confusion_matrix(y_true, y_pred)

        TN, FP, FN, TP = cm.ravel()

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = sensitivity  # Recall is the same as sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Print the metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall/Sensitivity: {recall:.2f}")
        print(f"F1-Score: {f1_score:.2f}")
        print(f"Specificity: {specificity:.2f}")

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['CT surveillance', 'Treatment'], yticklabels=['No LC', 'NSCLC'])
        plt.xlabel('Predicted outcomes')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for BTS Guideline Outcomes\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
        plt.show()
        return y_pred

    def apply_guideline_compare(self):
        """Apply the guideline logic to each patient for Brock and Herder models separately and compare with diagnosis."""
        brock_outcomes = []  # to store the outcome for each patient based on the Brock model
        herder_outcomes = []  # to store the outcome for each patient based on the Herder model

        for index, row in self.df.iterrows():
            # Initialize variables for Brock and Herder scores
            brock_score = row['% NSCLC in TM-model']
            herder_score = row['% LC in TM-model']
            nodule_size = row['Nodule size (1-30 mm)'] # Determine outcomes based on the Brock model score and nodule size
            if nodule_size < 8:
                brock_outcome = 'CT surveillance'
            elif nodule_size >= 8:
                if brock_score < 10:
                    brock_outcome = 'CT surveillance'
                elif 10 <= brock_score < 70:
                    brock_outcome = 'Consider image-guided biopsy'
                else:
                    brock_outcome = 'Consider excision or non-surgical treatment'
            else:
                brock_outcome = 'CT surveillance'


            # Determine outcomes based on the Herder model score and nodule size
            if nodule_size < 8:
                herder_outcome = 'CT surveillance'
            elif nodule_size >= 8:
                if herder_score < 10:
                    herder_outcome = 'CT surveillance'
                elif 10 <= herder_score < 70:
                    herder_outcome = 'Consider image-guided biopsy'
                else:
                    herder_outcome = 'Consider excision or non-surgical treatment'
            else:
                herder_outcome = 'CT surveillance'

            brock_outcomes.append(brock_outcome)
            herder_outcomes.append(herder_outcome)

        self.df['Brock Guideline Outcome'] = brock_outcomes
        self.df['Herder Guideline Outcome'] = herder_outcomes

        # Generate detailed outcomes dataframes with counts for each combination of outcome and diagnosis
        brock_detailed_outcomes = self.df.groupby(['Brock Guideline Outcome', 'Diagnose']).size().unstack(fill_value=0)
        herder_detailed_outcomes = self.df.groupby(['Herder Guideline Outcome', 'Diagnose']).size().unstack(fill_value=0)

        # Calculate total counts for each outcome
        brock_outcome_counts = self.df['Brock Guideline Outcome'].value_counts()
        herder_outcome_counts = self.df['Herder Guideline Outcome'].value_counts()

        # Print the detailed outcomes dataframes
        print("Brock Model Outcomes:")
        print(brock_detailed_outcomes)
        print("\nHerder Model Outcomes:")
        print(herder_detailed_outcomes)

        return (brock_detailed_outcomes, brock_outcome_counts), (herder_detailed_outcomes, herder_outcome_counts)

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
tree = DecisionTree(filepath)
results = tree.apply_guideline_logic()
# brock_results, herder_results = tree.apply_guideline_compare()
# tree.plot_confusion_matrix()
# tree.plot_roc_curve()
# tree.evaluate_metrics_cv()
