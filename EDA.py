import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import (chi2_contingency, fisher_exact, kruskal, mannwhitneyu,
                         spearmanr)
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class ModelEDA:

    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.protein_markers = ['CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA']
       #  self.categorical_vars = ['Current/Former smoker',
                        # 'Previous History of Extra-thoracic Cancer', 'Emphysema',
                        # 'Nodule Type', 'Nodule Upper Lobe', 'Nodule Count',
                        # 'Spiculation', 'PET-CT Findings']
        self.categorical_vars = ['Current/Former smoker', 'Emphysema', 'Spiculation', 'PET-CT Findings']

    def display_dataset_info(self):
        """Display dataset information in a formatted manner."""
        print("Dataset Information:\n")
        self.df.info()
        print("\n" + "="*60 + "\n")


    def display_summary_statistics(self):
        """Display summary statistics for the dataset."""
        print("Summary Statistics:\n")
        print(self.df[self.models].describe())

    def plot_protein_markers_distribution(self):
        """Plots the distribution of specified protein markers by LC diagnosis with individual plots for each marker."""
        for marker in self.protein_markers:
            # Filter data for the current marker
            data_filtered = self.df[['Diagnosis_Encoded', marker]].copy()
            data_filtered.rename(columns={marker: 'Level'}, inplace=True)
            data_filtered['Marker'] = marker  # Add a column for the marker name

            plt.figure(figsize=(10, 6))
            sns.violinplot(x='Marker', y='Level', hue='Diagnosis_Encoded', data=data_filtered, inner=None, split=True, palette="muted")
            sns.swarmplot(x='Marker', y='Level', hue='Diagnosis_Encoded', data=data_filtered, alpha=0.5, dodge=True)

            plt.title(f'Distribution of {marker} by Lung Cancer Diagnosis with Data Points')
            plt.xlabel('Protein Marker')
            plt.ylabel('Marker Level (ng/ml)')

            # Handling the legend to avoid duplicates and correctly label the categories
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = ['No LC', 'LC']
            plt.legend(handles=handles[0:2], labels=new_labels, title='Diagnosis')

            plt.tight_layout()
            plt.show()

    def plot_distributions_model_score(self):
        """Plot distributions for Brock score, Herder score, and TM percentages."""
        for model in self.models:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[model], kde=True, bins=20, edgecolor='k')
            plt.title(f'Distribution of {model}')
            plt.xlabel(model)
            plt.ylabel('Frequency')
            plt.show()

    def plot_categorical_frequency_with_diagnosis(self):
        """Plot frequency of each categorical variable for each diagnosis category."""
        for var in self.categorical_vars:
            plt.figure(figsize=(12, 8))

            # Create the count plot
            ax = sns.countplot(x='Diagnose', hue=var, data=self.df, palette='viridis')

            # Annotate plot with counts
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10), textcoords='offset points')

            plt.title(f'Count of {var} by Diagnosis')
            plt.xlabel('Diagnosis')
            plt.ylabel('Count')

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
        """Generate scatter plots overlayed with density plots for each model against nodule size, differentiated by categorical variable using shapes."""
        for model in self.models:
            for var in self.categorical_vars:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Use valid_data which has dropped NaN values for the current model and categorical variables
                valid_data = self.df.dropna(subset=['Nodule size (1-30 mm)', model, var])

                # KDE plot overlay
                sns.kdeplot(x='Nodule size (1-30 mm)', y=model, data=valid_data, fill=True,
                            levels=5, alpha=0.7, color='grey', ax=ax)

                # Define marker styles based on 'Spiculation' values
                # You can define more markers if you have more categories
                marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']

                # Assign a unique marker to each value of the categorical variable
                unique_values = valid_data[var].unique()
                markers = {value: marker for value, marker in zip(unique_values, marker_styles)}

                # Scatter plot with different shapes for 'Spiculation'
                sns.scatterplot(x='Nodule size (1-30 mm)', y=model, data=valid_data,
                                style=var, hue=var, markers=markers, edgecolor="none",
                                legend='full', s=100, ax=ax)  # Increased s for better visibility

                # Calculate Spearman correlation coefficient
                rho, p_value = spearmanr(valid_data['Nodule size (1-30 mm)'], valid_data[model])

                # Annotate the plot with Spearman rho value
                plt.annotate(f"Spearman ρ = {rho:.2f} (p = {p_value:.3f})", xy=(0.5, 0.95),
                             xycoords='axes fraction', ha='center', fontsize=10,
                             backgroundcolor='white')

                plt.title(f'Scatter and Density Plot of {model} vs. Nodule Size by {var}')
                plt.xlabel('Nodule size (1-30 mm)')
                plt.ylabel(f'{model} Score')
                plt.legend(title=var)
                plt.tight_layout()  # Adjust the layout to make room for the legend
                plt.show()

    def plot_model_score_vs_nodule_size_by_diagnosis(self):
        """Generate scatter plots overlayed with density plots for each model against nodule size, differentiated by diagnosis."""
        for model in self.models:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use valid_data which has dropped NaN values for the current model and 'Diagnose'
            valid_data = self.df.dropna(subset=['Nodule size (1-30 mm)', model, 'Diagnose'])

            # KDE plot overlay
            sns.kdeplot(x='Nodule size (1-30 mm)', y=model, data=valid_data, fill=True,
                        levels=5, alpha=0.7, color='grey', ax=ax)

            # Scatter plot with points colored by 'Diagnose'
            sns.scatterplot(x='Nodule size (1-30 mm)', y=model, data=valid_data,
                            hue='Diagnose', style='Diagnose',
                            markers={'No LC': 'o', 'NSCLC': 's'},  # Define markers for each diagnosis
                            edgecolor="none", s=100, ax=ax)  # Increased s for better visibility

            # Calculate Spearman correlation coefficient
            rho, p_value = spearmanr(valid_data['Nodule size (1-30 mm)'], valid_data[model])

            # Annotate the plot with Spearman rho value
            plt.annotate(f"Spearman ρ = {rho:.2f} (p = {p_value:.3f})", xy=(0.5, 0.95),
                         xycoords='axes fraction', ha='center', fontsize=10,
                         backgroundcolor='white')

            plt.title(f'Scatter and Density Plot of {model} vs. Nodule Size by Diagnosis')
            plt.xlabel('Nodule size (1-30 mm)')
            plt.ylabel(f'{model} Score')
            plt.legend(title='Diagnose')
            plt.tight_layout()  # Adjust the layout to make room for the legend
            plt.show()

    def plot_roc_curves(self):
        """
        Plots the ROC curve and calculates the AUC for each model's scores against the binary encoded diagnosis outcome.
        """
        for model in self.models:
            # Ensure the data is clean and contains no NaN values for both the model scores and the encoded diagnosis
            clean_df = self.df.dropna(subset=[model, 'Diagnosis_Encoded'])

            # Extract the model scores and the true binary encoded diagnosis outcomes
            scores = clean_df[model]
            true_outcomes = clean_df['Diagnosis_Encoded']

            # Calculate the ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(true_outcomes, scores)
            auc = roc_auc_score(true_outcomes, scores)

            # Plotting the ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label=f'{model} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Chance')  # Dashed diagonal for reference
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model}')
            plt.legend(loc='lower right')
            plt.show()

    def plot_confusion_matrices(self, threshold=50):
        """
        Plots confusion matrices and calculates sensitivity and specificity for each model based on a specified threshold.

        Parameters:
        - threshold: The threshold for predicting the positive class (default is 10% or 0.1).
        """
        for model in self.models:
            # Ensure there are no NaN values for the model scores and diagnosis
            clean_df = self.df.dropna(subset=[model, 'Diagnosis_Encoded'])

            # Apply threshold to model scores to generate binary predictions
            predictions = (clean_df[model] >= threshold).astype(int)

            # Generate the confusion matrix
            cm = confusion_matrix(clean_df['Diagnosis_Encoded'], predictions)

            # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate)
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]

            sensitivity = TP / (TP + FN) if (TP + FN) else 0
            specificity = TN / (TN + FP) if (TN + FP) else 0

            # Plotting the confusion matrix
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues',
                        xticklabels=['No LC', 'NSCLC'], yticklabels=['No LC', 'NSCLC'])
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.title(f'Confusion Matrix for {model}\nThreshold: {threshold}%\n'
                      f'Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
            plt.show()

    def calculate_best_sensitivity_specificity(self):
        """
        Calculates the optimal threshold for each model based on clinical decision analysis.
        """
        clinical_values = {
            'TP': -1,  # Cost/benefit of a true positive (e.g., early treatment)
            'FP': -5,  # Cost of a false positive (e.g., unnecessary treatment)
            'TN': 0,   # Benefit of a true negative
            'FN': -20  # Cost of a false negative (e.g., missed diagnosis)
        }

        optimal_results = {}
        for model in self.models:
            clean_df = self.df.dropna(subset=[model, 'Diagnosis_Encoded'])
            thresholds = sorted(clean_df[model].unique())
            best_clinical_value = float('inf')
            best_threshold = None

            for threshold in thresholds:
                predictions = (clean_df[model] >= threshold).astype(int)
                cm = confusion_matrix(clean_df['Diagnosis_Encoded'], predictions)
                TP = cm[1, 1]
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]

                # Calculate the clinical value for the current threshold
                current_clinical_value = (
                                        TP * clinical_values['TP'] +
                                        FP * clinical_values['FP'] +
                                        TN * clinical_values['TN'] +
                                        FN * clinical_values['FN']
                                        )
                # If the current clinical value is less than the best (since costs are negative, we want the least negative value),
                # update the best clinical value and corresponding threshold
                if current_clinical_value < best_clinical_value:
                    best_clinical_value = current_clinical_value
                    best_threshold = threshold

            # Assuming that lower costs are better, find the threshold with the minimum cost
            optimal_results[model] = {
                'Optimal Threshold': best_threshold,
                'Clinical Value': best_clinical_value
            }
        print(optimal_results)
        return optimal_results

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

    def calculate_vif_for_protein_markers(self):
        """
        Calculates Variance Inflation Factors (VIF) for each protein marker in the DataFrame.
        Returns:
        - A pandas DataFrame containing the VIF for each protein marker.
        """
        # Ensure only valid protein markers present in the DataFrame are considered
        valid_protein_markers = [marker for marker in self.protein_markers if marker in self.df.columns]

        # Filter the DataFrame to include only the valid protein markers plus a constant term for the intercept
        X = add_constant(self.df[valid_protein_markers])

        # Initialize a DataFrame to store VIF results
        vif_data = pd.DataFrame()
        vif_data["Protein Marker"] = X.columns.drop('const')  # Exclude the constant term from the results
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(1, len(X.columns))]  # Skip the first index for 'const'

        return vif_data

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
eda = ModelEDA(filepath)
#eda.display_dataset_info()
#eda.display_summary_statistics()
#eda.plot_distributions_model_score()
#eda.plot_categorical_frequency_with_diagnosis()
#eda.plot_relationship_with_stadium()
#eda.plot_stadium_frequency()
# eda.plot_node_size_distributions()
eda.plot_protein_markers_distribution()
#eda.plot_model_score_vs_nodule_size()
#eda.plot_model_score_vs_nodule_size_by_diagnosis()
#eda.plot_roc_curves()
#eda.plot_confusion_matrices()
#eda.calculate_best_sensitivity_specificity()
# eda.test_significance_of_categorical_variables_with_model()
# eda.test_protein_marker_significance_with_model()
# eda.test_significance_of_categorical_variables_with_diagnosis()
# eda.test_protein_marker_significance_with_diagnosis()
# vif_results=eda.calculate_vif_for_protein_markers()
# print(vif_results)
