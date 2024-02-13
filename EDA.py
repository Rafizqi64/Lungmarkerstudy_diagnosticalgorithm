import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, kruskal

class ModelEDA:
    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
    
    def display_dataset_info(self):
        """Display dataset information in a formatted manner."""
        print("Dataset Information:\n")
        self.df.info()
        print("\n" + "="*60 + "\n")

    def display_summary_statistics(self):
        """Display summary statistics for the dataset."""
        print("Summary Statistics:\n")
        print(self.df[self.models].describe())
        
    def plot_distributions_model_score(self):
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
            ax = sns.stripplot(x=stadium_series, y=model, data=self.df, order=unique_stages)
      
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
    
    def plot_protein_marker_correlations(self):
        """Plot a heatmap of the correlation matrix for all protein markers with LC and NSCLC values."""
        # Select only the columns related to protein markers and model scores
        data_subset = self.df[self.protein_markers + self.models]
        
        # Compute the correlation matrix
        corr_matrix = data_subset.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Protein Markers and LC/NSCLC Values')
        plt.show()

    def test_protein_marker_significance(self):
        """Calculate and print the Pearson correlation coefficient and p-value for each protein marker and model score."""
        print("Testing correlation and significance between protein markers and model scores:\n" + "="*60)
        for model in self.models:
            print(f"\nTesting significance for model: {model}\n" + "="*60)
            for marker in self.protein_markers:
                # Ensure no NaN values are present in the series
                clean_marker = self.df[marker].dropna()
                clean_model_scores = self.df[model].dropna()

                # Only proceed if there are enough values to calculate the correlation
                if len(clean_marker) == len(clean_model_scores) and len(clean_marker) > 1:
                    corr, p_val = pearsonr(clean_marker, clean_model_scores)
                    significance, color = ("Significant", "\033[92m") if p_val < 0.05 else ("Not significant", "\033[91m")
                
                    print(f"{marker} and {model}: Coefficient={corr:.2f}, P-value={p_val:.3f} {color}({significance})\033[0m")
                else:
                    print(f"Not enough data to calculate correlation between {marker} and {model}.")

    def plot_node_size_distributions(self):
        """Visualize the distribution of numerical variables."""
        numerical_vars = ['Nodule size (1-30 mm)']
        for var in numerical_vars:
            sns.histplot(self.df[var].dropna(), kde=True)
            plt.title(f'Distribution of {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.show()

    def plot_binary_distributions(self):
        """Visualize the distribution of categorical variables."""
        categorical_vars = ['Family History of LC', 'Current/Former smoker', 'Previous History of Extra-thoracic Cancer', 'Emphysema', 'Nodule Upper Lobe', 'Spiculation']
          # Melt the dataframe so all values are in one column
        melted_df = self.df[categorical_vars].melt(var_name='Variables', value_name='Values')
        
        plt.figure(figsize=(10, 8))
        ax = sns.countplot(x='Values', hue='Variables', data=melted_df, palette='Set2')
        ax.set_title('Count of Categories Across Binary Variables')
        plt.xticks(rotation=90)
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
        plt.show()

    def plot_categorical_relationship_with_models(self):
        """Visualize the relationship between categorical variables and model scores."""
        for model in self.models:
            for var in ['Current/Former smoker', 
                            'Previous History of Extra-thoracic Cancer', 'Emphysema', 
                            'Nodule Type', 'Nodule Upper Lobe', 'Nodule Count', 
                            'Spiculation', 'PET-CT Findings']:
                sns.violinplot(x=self.df[var], y=self.df[model])
                plt.title(f'{model} Score by {var}')
                plt.xlabel(var)
                plt.ylabel(f'{model} Score')
                plt.xticks(rotation=45)
                plt.show()

    def test_significance_of_categorical_variables(self):
        """Test for the significance of differences in model scores across binary categorical variables
        using Mann-Whitney U test and multicoategorical variables with Kruskal-Wallis test."""
        categorical_vars=['Current/Former smoker', 
                            'Previous History of Extra-thoracic Cancer', 'Emphysema', 
                            'Nodule Type', 'Nodule Upper Lobe', 'Nodule Count', 
                            'Spiculation', 'PET-CT Findings']
         
        for model in self.models:
            print(f"\nTesting significance for model: {model}\n" + "="*60)
            for var in categorical_vars:
                categories = self.df[var].dropna().unique()
                
                if len(categories) == 2:
                    # Binary categorical variable, use Mann-Whitney U test
                    group1 = self.df[self.df[var] == categories[0]][model].dropna()
                    group2 = self.df[self.df[var] == categories[1]][model].dropna()
                    u_stat, p_val = mannwhitneyu(group1, group2)
                    significance, color = ("Significant", "\033[92m") if p_val < 0.05 else ("Not significant", "\033[91m")
                    print(f"{var} (binary): Mann-Whitney U test: U-Statistic={u_stat}, P-value={p_val:.3f} {color}({significance})\033[0m")
                
                elif len(categories) > 2:
                    # Multi-categorical variable, use Kruskal-Wallis test
                    groups = [self.df[self.df[var] == category][model].dropna() for category in categories]
                    k_stat, p_val = kruskal(*groups)
                    significance, color = ("Significant", "\033[92m") if p_val < 0.05 else ("Not significant", "\033[91m")
                    print(f"{var} (multi-category): Kruskal-Wallis test: H-Statistic={k_stat}, P-value={p_val:.3f} {color}({significance})\033[0m")
                              
                else:
                    print(f"Not enough categories to test for {var}.")

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
eda = ModelEDA(filepath)
eda.display_dataset_info()
eda.display_summary_statistics()
#eda.plot_distributions_model_score()
#eda.plot_relationship_with_diagnosis()
#eda.correlation_with_diagnosis()
#eda.plot_relationship_with_stadium()
#eda.plot_stadium_frequency()
eda.test_protein_marker_significance()
# eda.plot_node_size_distributions()
# eda.plot_binary_distributions()
eda.plot_categorical_relationship_with_models()
eda.test_significance_of_categorical_variables()
