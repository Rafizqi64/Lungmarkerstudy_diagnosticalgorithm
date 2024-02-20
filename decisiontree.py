import numpy as np
import pandas as pd


class DecisionTree:

    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
        self.categorical_vars = ['Current/Former smoker', 'Emphysema', 'Spiculation', 'PET-CT Findings']
    
    def apply_guideline_logic_LBx(self):
        """Apply the guideline logic to each patient and compare with diagnosis."""
        outcomes = []  # to store the outcome for each patient

        for index, row in self.df.iterrows():
            # Initialize variables for LBx scores
            lc_score = row['% LC in TM-model']
            nsclc_score = row['% NSCLC in TM-model']
            if row['Nodule size (1-30 mm)'] < 5:
                outcome = 'Discharge'
            elif row['Nodule size (1-30 mm)'] < 8:
                outcome = 'CT surveillance'
            elif row['Nodule size (1-30 mm)'] >= 8:
                if lc_score < 10:
                    outcome = 'CT surveillance'
                else:
                    if nsclc_score < 10:
                        outcome = 'CT surveillance'
                    elif 10 <= nsclc_score < 70:
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
 
filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
tree = DecisionTree(filepath)
results = tree.apply_guideline_logic_LBx()
results2 = tree.apply_guideline_logic()
