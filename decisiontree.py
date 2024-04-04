import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DecisionTree:

    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
        self.categorical_vars = ['Current/Former smoker', 'Emphysema', 'Spiculation', 'PET-CT Findings']
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

            # Append the outcome to the outcomes list
            outcomes.append(outcome)

        # Add the outcomes to the DataFrame
        self.df['Guideline Outcome'] = outcomes

        # Generate a detailed outcomes dataframe with counts for each combination of outcome and diagnosis
        detailed_outcomes = self.df.groupby(['Guideline Outcome', 'Diagnose']).size().unstack(fill_value=0)

        # Calculate total counts for each outcome
        outcome_counts = self.df['Guideline Outcome'].value_counts()
        print(detailed_outcomes)
        # Return the detailed outcomes dataframe and the outcome counts
        return detailed_outcomes, outcome_counts
    def plot_confusion_matrix(self):
        outcome_mapping = {
            'Discharge': 0,
            'CT surveillance': 0,
            'Consider image-guided biopsy': 1,
            'Consider excision or non-surgical treatment': 1,
        }
        self.df['Predicted'] = self.df['Guideline Outcome'].map(outcome_mapping)

        # True labels, already encoded as 0 for 'No LC' and 1 for 'NSCLC'
        y_true = self.df['Diagnosis_Encoded']

        # Predicted labels, as mapped above
        y_pred = self.df['Predicted']

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Extracting TN, FP, FN, TP
        TN, FP, FN, TP = cm.ravel()

        # Calculating sensitivity and specificity
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # Calculating metrics
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

        # Add the outcomes to the DataFrame
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

        # Return the detailed outcomes dataframes and the outcome counts
        return (brock_detailed_outcomes, brock_outcome_counts), (herder_detailed_outcomes, herder_outcome_counts)

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
tree = DecisionTree(filepath)
results = tree.apply_guideline_logic()
# brock_results, herder_results = tree.apply_guideline_compare()
tree.plot_confusion_matrix()
#results2 = tree.apply_guideline_logic()
