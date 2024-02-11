import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEDA:
    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
    def display_dataset_info(self):
        """Display dataset information in a formatted manner."""
        print("Dataset Information:\n")
        self.df.info()
        print("\n" + "="*60 + "\n")

    def display_summary_statistics(self):
        """Display summary statistics for the dataset."""
        print("Summary Statistics:\n")
        print(self.df[self.models].describe())
        
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
            ax = sns.boxplot(x='Diagnose', y=model, data=self.df)
            ax.set_ylim(0, 100)  # Standardize the y-axis from 0 to 100
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

    def plot_relationship_with_stadium(self):
        """Plot the relationship of each model score with the cancer stages, handling NaN values and mixed types."""
        for model in self.models:
            plt.figure(figsize=(10, 6))
            
            # Clean 'Stadium' column, convert to string, and handle NaN values
            stadium_series = self.df['Stadium'].fillna('No Stadium').astype(str)
            
            # Get the unique values and sort them, excluding 'No Stadium'
            unique_stages = sorted(stadium_series.unique(), key=lambda x: (x.isdigit(), x))
            
            # Move 'No Stadium' to the end of the list if it exists
            if 'No Stadium' in unique_stages:
                unique_stages.append(unique_stages.pop(unique_stages.index('No Stadium')))
            
            # Now create the boxplot with the sorted, cleaned list
            ax = sns.boxplot(x=stadium_series, y=model, data=self.df, order=unique_stages)
      
            ax.set_title(f'Relationship of {model} with Cancer Stages')
            ax.set_ylim(0, 103) 
            ax.set_xlabel('Cancer Stage')
            ax.set_ylabel(f'{model} Score')
            plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability if necessary
            plt.show()

    def plot_stadium_frequency(self):
        """Plot the frequency of each cancer stage, handling NaN values and the '0' category."""
        plt.figure(figsize=(10, 6))
        
        # Clean 'Stadium' column, handle NaN values and convert all to string for consistency
        stadium_series = self.df['Stadium'].fillna('No Stadium').astype(str)
        
        # Convert '0' to 'Stage 0' to distinguish it as a category
        stadium_series = stadium_series.replace('0', 'Stage 0')

        # Calculate the frequency of each category
        stadium_counts = stadium_series.value_counts().sort_index()

        # Plot a bar chart
        stadium_counts.plot(kind='bar')
        plt.title('Frequency of Each Cancer Stage')
        plt.xlabel('Cancer Stage')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.show()

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
eda = ModelEDA(filepath)
eda.display_dataset_info()  # Dataset information
eda.display_summary_statistics() # Summary statistics
#eda.plot_distributions()  # Distribution plots for each model score
#eda.plot_relationship_with_diagnosis()  # Relationship of model scores with diagnosis
#eda.correlation_with_diagnosis()  # Correlation of model scores with diagnosis
#eda.plot_relationship_with_stadium()
eda.plot_stadium_frequency()
