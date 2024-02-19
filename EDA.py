import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import (chi2_contingency, fisher_exact, kruskal, mannwhitneyu,
                         spearmanr)


class ModelEDA:

    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
        self.categorical_vars = ['Current/Former smoker', 
                        'Previous History of Extra-thoracic Cancer', 'Emphysema', 
                        'Nodule Type', 'Nodule Upper Lobe', 'Nodule Count', 
                        'Spiculation', 'PET-CT Findings']

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
        """Plot the relationship of each model score with the diagnosis, coloring by category, and displaying class counts for each diagnosis right under the groups."""
        for model in self.models:
            for var in self.categorical_vars: 
                plt.figure(figsize=(12, 8))
                
                # Create the strip plot
                ax = sns.stripplot(x='Diagnose', y=model, hue=var, data=self.df, dodge=True)
                
                # Calculate counts for each class within each diagnosis category
                class_counts_by_diag = self.df.groupby(['Diagnose', var]).size().unstack(fill_value=0)
                
                # Lower the bottom of the y-axis to make space for annotations
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin - (0.1 * ymax), ymax)
                
                # Annotate plot with counts right under each group
                for i, diag in enumerate(self.df['Diagnose'].unique()):
                    for j, category in enumerate(self.df[var].unique()):
                        count = class_counts_by_diag.loc[diag, category] if diag in class_counts_by_diag.index and category in class_counts_by_diag.columns else 0
                        # Adjust x positioning based on number of categories and dodge
                        x = i + (j - len(self.df[var].unique())/2)*0.1  
                        plt.text(x, ymin - (0.05 * ymax), f'{count}', ha='center', va='top', fontsize=9)
                
                plt.title(f'{model} by Diagnosis and {var}')
                plt.xlabel('Diagnosis')
                plt.ylabel(model)
                
                # Move the legend outside the plot
                plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.show()


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
        plt.xticks(rotation=45)
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


    def plot_node_size_distributions(self):
        """Visualize the distribution of numerical variables."""
        numerical_vars = ['Nodule size (1-30 mm)']
        plt.show()
        for var in numerical_vars:
            sns.histplot(self.df[var].dropna(), kde=True)
            plt.title(f'Distribution of {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.show()


    def plot_model_score_vs_nodule_size(self):
        """Generate scatter plots overlayed with density plots for each model against nodule size."""
        for model in self.models:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use valid_data which has dropped NaN values for the current model
            valid_data = self.df.dropna(subset=['Nodule size (1-30 mm)', model])

            # Scatter plot
            sns.scatterplot(x='Nodule size (1-30 mm)', y=model, data=valid_data,
                            edgecolor="none", legend=False, s=10, ax=ax, color='blue')

            # KDE plot overlay
            sns.kdeplot(x='Nodule size (1-30 mm)', y=model, data=valid_data, fill=True,
                        levels=5, alpha=0.7, color='blue', ax=ax)

            # Calculate Spearman correlation coefficient
            rho, p_value = spearmanr(valid_data['Nodule size (1-30 mm)'], valid_data[model])

            # Annotate the plot with Spearman rho value
            plt.annotate(f"Spearman ρ = {rho:.2f} (p = {p_value:.3f})", xy=(0.5, 0.95), xycoords='axes fraction', 
                         ha='center', fontsize=10, backgroundcolor='white')

            plt.title(f'Scatter and Density Plot of {model} vs. Nodule Size')
            plt.xlabel('Nodule size (1-30 mm)')
            plt.ylabel(f'{model} Score')
            plt.show()


    def plot_categorical_relationship_with_models(self):
        """Visualize the relationship between categorical variables and model scores."""
        for model in self.models:
            for var in self.categorical_vars: 
                plt.title(f'{model} Score by {var}')
                plt.xlabel(var)
                plt.ylabel(f'{model} Score')
                plt.xticks(rotation=45)
                plt.show()

    def test_protein_marker_significance_with_diagnosis(self):
        """Test the association between protein markers and binary diagnosis using logistic regression."""
        print("Testing association between protein markers and diagnosis:\n" + "="*60)
       
        # Encode the diagnosis variable: 'No LC' to 0, 'NSCLC' to 1
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)

        for marker in self.protein_markers:
            print(f"\nTesting association for marker: {marker}\n" + "="*60)
            
            # Prepare the data: drop rows with NaN in either the marker or the encoded diagnosis
            data = self.df.dropna(subset=[marker, 'Diagnosis_Encoded'])
            
            # Logistic regression
            X = sm.add_constant(data[marker])  # Adds a constant term to the predictor
            y = data['Diagnosis_Encoded']
            
            try:
                model = sm.Logit(y, X).fit(disp=0)  # Fit the model without printing the summary
                
                # Extract the p-value for the marker
                p_val = model.pvalues[marker]
                
                # Print the odds ratio for easier interpretation of the effect size
                odds_ratio = np.exp(model.params[marker])
                
                significance = "\033[92mSignificant\033[0m" if p_val < 0.05 else "\033[91mNot significant\033[0m"
                print(f"Marker: {marker}, Odds Ratio={odds_ratio:.4f}, P-value={p_val:.3f} ({significance})")
            except Exception as e:
                print(f"Could not calculate association for {marker}. Reason: {str(e)}")

    def test_protein_marker_significance_with_model(self):
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
                    corr, p_val = spearmanr(clean_marker, clean_model_scores)
                    significance, color = ("Significant", "\033[92m") if p_val < 0.05 else ("Not significant", "\033[91m")
                    print(f"{marker} and {model}: Coefficient={corr:.2f}, P-value={p_val:.3f} {color}({significance})\033[0m")
                else:
                    print(f"Not enough data to calculate correlation between {marker} and {model}.")


    def test_significance_of_categorical_variables_with_diagnosis(self):
        """Test the significance of categorical variables against the diagnosis using Chi-Square and Fisher's Exact tests."""
        diagnosis_var = 'Diagnose'
        for var in self.categorical_vars:
            print(f"\nTesting significance for {var} with {diagnosis_var}:\n" + "="*60)
            contingency_table = pd.crosstab(self.df[diagnosis_var], self.df[var])

            if contingency_table.shape[1] == 2:
                # Binary categorical variable, use Fisher's Exact Test
                _, p_val = fisher_exact(contingency_table)
                test_used = "Fisher's Exact"
            else:
                # Multi-categorical variable, use Chi-Square Test
                _, p_val, _, _ = chi2_contingency(contingency_table)
                test_used = "Chi-Square"
                
            significance = "\033[92mSignificant\033[0m" if p_val < 0.05 else "\033[91mNot significant\033[0m"
            print(f"{test_used} test for {var}: P-value={p_val:.3f} ({significance})")


    def test_significance_of_categorical_variables_with_model(self):
        """Test for the significance of differences in model scores across binary categorical variables
        using Mann-Whitney U test and multicoategorical variables with Kruskal-Wallis test."""
        
        for model in self.models:
            print(f"\nTesting significance for model: {model}\n" + "="*60)
            for var in self.categorical_vars:
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
#eda.display_dataset_info()
#eda.display_summary_statistics()
#eda.plot_distributions_model_score()
#eda.plot_relationship_with_diagnosis()
#eda.plot_relationship_with_stadium()
#eda.plot_stadium_frequency()
# eda.plot_node_size_distributions()
# eda.plot_model_score_vs_nodule_size()
# eda.plot_categorical_relationship_with_models()
# eda.test_significance_of_categorical_variables_with_model()
#eda.test_protein_marker_significance_with_model()
eda.test_significance_of_categorical_variables_with_diagnosis()
eda.test_protein_marker_significance_with_diagnosis()
