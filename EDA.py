import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEDA:
    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.models = ['Brock score', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']

    def plot_distributions(self):
        """Plot distributions for Brock score, Herder score, and TM percentages."""
        for model in self.models:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[model], kde=True, bins=20, edgecolor='k')
            plt.title(f'Distribution of {model}')
            plt.xlabel(model)
            plt.ylabel('Frequency')
            plt.show()

    def plot_relationship_with_diagnosis(self):
        """Plot the relationship of each model score with the diagnosis."""
        for model in self.models:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Diagnose', y=model, data=self.df)
            plt.title(f'{model} by Diagnosis')
            plt.xlabel('Diagnosis')
            plt.ylabel(model)
            plt.show()

    def correlation_with_diagnosis(self):
        """Compute and display correlation of model scores with diagnosis (numerically encoded if necessary)."""
        # Encode the 'Diagnose' column if it's categorical (No LC, NSCLC, etc.)
        if self.df['Diagnose'].dtype == 'object':
            diagnosis_mapping = {'No LC': 0, 'NSCLC': 1}  # Example mapping, adjust based on actual diagnoses
            self.df['Diagnose_Encoded'] = self.df['Diagnose'].map(diagnosis_mapping)
            target = 'Diagnose_Encoded'
        else:
            target = 'Diagnose'

        # Calculate and print correlation
        correlations = self.df[self.models + [target]].corr()[target].sort_values(ascending=False)
        print("Correlation of models with diagnosis:\n", correlations)

# Usage
filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
eda = ModelEDA(filepath)
eda.plot_distributions()  # Distribution plots for each model score
eda.plot_relationship_with_diagnosis()  # Relationship of model scores with diagnosis
eda.correlation_with_diagnosis()  # Correlation of model scores with diagnosis

