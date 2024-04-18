import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import (chi2_contingency, fisher_exact, kruskal, mannwhitneyu,
                         spearmanr)
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class ModelEDA:
    def __init__(self, filepath):
        """Load the dataset and initialize."""
        self.df = pd.read_excel(filepath)
        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)
        self.models = ['Brock score (%)', 'Herder score (%)', '% LC in TM-model', '% NSCLC in TM-model']
        self.numerical_vars = ['Nodule size (1-30 mm)', 'Nodule Count', 'CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE corrected for H-index', 'proGRP', 'SCCA']
        self.categorical_vars = ['Current/Former smoker',
                        'Family History of LC', 'Emphysema',
                        'Nodule Type', 'Nodule Upper Lobe',
                        'Spiculation', 'PET-CT Findings']
        # self.categorical_vars = ['Current/Former smoker', 'Emphysema', 'PET-CT Findings']

    def preprocess_data_for_table(self):
        """Preprocess the dataset."""
        binary_map = {'Nee': 0, 'Ja': 1}
        self.df['Current/Former smoker'] = self.df['Current/Former smoker'].replace(binary_map)
        self.df['Family History of LC'] = self.df['Family History of LC'].replace(binary_map)
        self.df['Emphysema'] = self.df['Emphysema'].replace(binary_map)
        self.df['Nodule Upper Lobe'] = self.df['Nodule Upper Lobe'].replace(binary_map)
        self.df['Spiculation'] = self.df['Spiculation'].replace(binary_map)
        one_hot_vars = []
        for var in ['Nodule Type', 'PET-CT Findings']:
            dummies = pd.get_dummies(self.df[var], prefix=var)
            self.df.drop(var, axis=1, inplace=True)  # Remove the original column
            self.df = pd.concat([self.df, dummies], axis=1)
            # Update the categorical_vars list with the new one-hot encoded variables
            one_hot_vars.extend(dummies.columns.tolist())

        # Update categorical_vars to include new one-hot encoded columns and exclude the original columns
        self.categorical_vars = [var for var in self.categorical_vars if var not in ['Nodule Type', 'PET-CT Findings']] + one_hot_vars
        return self.df

    def preprocess_data(self):
        binary_map = {'Nee': 0, 'Ja': 1}
        self.df['Current/Former smoker'] = self.df['Current/Former smoker'].replace(binary_map)
        self.df['Family History of LC'] = self.df['Family History of LC'].replace(binary_map)
        self.df['Emphysema'] = self.df['Emphysema'].replace(binary_map)
        self.df['Nodule Upper Lobe'] = self.df['Nodule Upper Lobe'].replace(binary_map)
        self.df['Spiculation'] = self.df['Spiculation'].replace(binary_map)

        # One-hot encoding for Nodule Type
        nodule_types = pd.get_dummies(self.df['Nodule Type'], prefix='Nodule_type')
        self.df = pd.concat([self.df, nodule_types], axis=1)
        return self.df

    def calculate_distribution(self, var):
        if var in ['Current/Former smoker', 'Family History of LC', 'Emphysema', 'Nodule Upper Lobe', 'Spiculation']:
            # For original binary variables, calculate the distribution as percentage for both 1 (Yes) and 0 (No) values
            distribution = self.df.groupby('Diagnose')[var].value_counts(normalize=True).unstack(fill_value=0) * 100
        elif var.startswith('Nodule Type_') or var.startswith('PET-CT Findings_'):
            # For one-hot encoded binary variables, directly calculate the mean as the distribution
            distribution = self.df.groupby('Diagnose')[var].mean() * 100
        else:
            # For numerical variables, calculate mean and standard deviation for each diagnosis group
            distribution = self.df.groupby('Diagnose')[var].agg(['mean', 'std'])
        return distribution

    def generate_summary_table(self):
        self.preprocess_data_for_table()
        summary_rows = []

        for var in self.categorical_vars + self.numerical_vars:
            distribution = self.calculate_distribution(var)

            if var in ['Current/Former smoker', 'Family History of LC', 'Emphysema', 'Nodule Upper Lobe', 'Spiculation']:
                # Original binary variables
                no_lc_dist = f"No: {distribution.loc['No LC', 0]:.2f}%, Yes: {distribution.loc['No LC', 1]:.2f}%"
                nsclc_dist = f"No: {distribution.loc['NSCLC', 0]:.2f}%, Yes: {distribution.loc['NSCLC', 1]:.2f}%"
            elif isinstance(distribution, pd.DataFrame) and 'mean' in distribution.columns:
                # Numerical variables
                no_lc_dist = f"Mean: {distribution.loc['No LC', 'mean']:.2f}, Std: {distribution.loc['No LC', 'std']:.2f}"
                nsclc_dist = f"Mean: {distribution.loc['NSCLC', 'mean']:.2f}, Std: {distribution.loc['NSCLC', 'std']:.2f}"
            else:
                # One-hot encoded binary variables
                no_lc_dist = f"{distribution.loc['No LC']:.2f}%"
                nsclc_dist = f"{distribution.loc['NSCLC']:.2f}%"

            p_val, test_used = self.perform_significance_testing(var, var not in self.categorical_vars)
            significance = "Significant" if p_val < 0.05 else "Not significant"

            summary_rows.append({
                'Variable': var,
                'P-value': p_val,
                'Significance': significance,
                'Test Used': test_used,
                'Distribution No LC': no_lc_dist,
                'Distribution NSCLC': nsclc_dist
            })

        summary_table = pd.DataFrame(summary_rows)
        print(summary_table)
        return summary_table

    def perform_significance_testing(self, var, is_numerical):
        if is_numerical:
            data = self.df.dropna(subset=[var, 'Diagnosis_Encoded'])
            X = sm.add_constant(data[var])
            y = data['Diagnosis_Encoded']
            try:
                model = sm.Logit(y, X).fit(disp=0)
                p_val = model.pvalues[var]
                test_used = "Logistic Regression"
            except Exception:
                p_val = np.nan  # Ensure a value is assigned even in case of an error
                test_used = "Error in Logistic Regression"
        else:
            contingency_table = pd.crosstab(self.df['Diagnose'], self.df[var])
            if contingency_table.shape[1] == 2:
                _, p_val = fisher_exact(contingency_table)
                test_used = "Fisher's Exact"
            else:
                _, p_val, _, _ = chi2_contingency(contingency_table)
                test_used = "Chi-Square"
        return p_val, test_used

    def evaluate_simple_model_scores(self, true_label_col, score_col, threshold=50, num_folds=5):
        """
        Evaluates the model scores with specified metrics using Stratified K-Fold cross-validation.

        Parameters:
        - df: DataFrame containing the dataset with true labels and scores.
        - true_label_col: The name of the column containing the true binary labels.
        - score_col: The name of the column containing the scores.
        - threshold: The threshold to convert scores into binary predictions (default is 50).
        - num_folds: Number of folds for Stratified K-Fold cross-validation (default is 5).

        Returns:
        - A dictionary with average values and standard deviations of Accuracy, Precision, Recall, F1 Score, ROC AUC, and Specificity.
        """
        y = self.df[true_label_col].values
        scores = self.df[score_col].values

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'specificity': []}

        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):  # Using dummy X since it's not used
            y_test = y[test_idx]
            scores_test = scores[test_idx]

            predictions = (scores_test >= threshold).astype(int)

            # Calculate metrics for this fold
            metrics['accuracy'].append(accuracy_score(y_test, predictions))
            metrics['precision'].append(precision_score(y_test, predictions))
            metrics['recall'].append(recall_score(y_test, predictions))
            metrics['f1'].append(f1_score(y_test, predictions))
            metrics['roc_auc'].append(roc_auc_score(y_test, scores_test))

            # Calculate specificity for this fold
            tn, fp, _, _ = confusion_matrix(y_test, predictions).ravel()
            specificity = tn / (tn + fp)
            metrics['specificity'].append(specificity)

        # Calculate the average and standard deviation of each metric across all folds
        avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        std_metrics = {metric: np.std(values) for metric, values in metrics.items()}
        metrics_output = {metric: {'average': avg, 'std_dev': std_metrics[metric]} for metric, avg in avg_metrics.items()}

        return metrics_output

    def plot_prediction_histogram(self, true_label_col, score_col, threshold):
        """
        Plots histograms of the predictions for positives and negatives, along with the threshold.

        Parameters:
        - score_col: The name of the column containing the model scores.
        - true_label_col: The name of the column containing the true labels.
        - threshold: The threshold value to categorize predictions.
        """
        scores = self.df[score_col]
        y = self.df[true_label_col]
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 7))

        # Plot distribution for negative and positive classes
        sns.histplot(scores[y == 0], bins=20, kde=True, label='No LC', color='blue', alpha=0.5)
        sns.histplot(scores[y == 1], bins=30, kde=True, label='NSCLC', color='red', alpha=0.7)

        # Plot threshold line
        plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.2f}')

        plt.title(f'Probability Distribution of {score_col} with Threshold {threshold:.2f}', fontsize=10)
        plt.xlabel('Probability of being Positive Class', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 100)
        plt.show()

    def display_dataset_info(self):
        """Display dataset information in a formatted manner."""
        print("Dataset Information:\n")
        self.df.info()
        print("\n" + "="*60 + "\n")


    def display_summary_statistics(self):
        """Display summary statistics for the dataset."""
        print("Summary Statistics:\n")
        print(self.df[self.models].describe())

    def plot_numerical_vars_distribution(self):
        """Plots the distribution of specified protein markers by LC diagnosis with individual plots for each marker."""
        for marker in self.numerical_vars:
            # Filter data for the current marker
            data_filtered = self.df[['Diagnosis_Encoded', marker]].copy()
            data_filtered.rename(columns={marker: 'Level'}, inplace=True)
            data_filtered['Marker'] = marker  # Add a column for the marker name

            plt.figure(figsize=(10, 6))
            sns.violinplot(x='Marker', y='Level', hue='Diagnosis_Encoded', data=data_filtered, inner=None, split=True, palette="muted")
            sns.swarmplot(x='Marker', y='Level', hue='Diagnosis_Encoded', data=data_filtered, alpha=0.5, dodge=True)

            plt.title(f'Distribution of {marker} by Lung Cancer Diagnosis with Data Points')
            plt.xlabel('Protein Marker')
            plt.ylabel('Marker Level (pg/ml)')

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

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10), textcoords='offset points')

            plt.title(f'Count of {var} by Diagnosis')
            plt.xlabel('Diagnosis')
            plt.ylabel('Count')

            plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.show()

    def plot_relationship_with_stadium(self):
        """Plot the relationship of each model score with the cancer stages, handling NaN values and mixed types."""
        for model in self.models:
            plt.figure(figsize=(10, 6))

            # Clean 'Stadium' column, convert to string, and handle NaN values
            stadium_series = self.df['Stadium'].fillna('No Stadium').astype(str)

            unique_stages = sorted(stadium_series.unique(), key=lambda x: (x.isdigit(), x))

            if 'No Stadium' in unique_stages:
                unique_stages.append(unique_stages.pop(unique_stages.index('No Stadium')))

            ax = sns.stripplot(x=stadium_series, y=model, data=self.df, order=unique_stages)

            ax.set_title(f'Relationship of {model} with Cancer Stages')
            ax.set_ylim(0, 103)
            ax.set_xlabel('Cancer Stage')
            ax.set_ylabel(f'{model} Score')
            plt.xticks(rotation=45)
            plt.show()


    def plot_stadium_frequency(self):
        """Plot the frequency of each cancer stage, handling NaN values and the '0' category."""
        plt.figure(figsize=(10, 6))

        # Clean 'Stadium' column, handle NaN values and convert all to string for consistency
        stadium_series = self.df['Stadium'].fillna('No Stadium').astype(str)

        stadium_series = stadium_series.replace('0', 'Stage 0')

        stadium_counts = stadium_series.value_counts().sort_index()

        stadium_counts.plot(kind='bar')
        plt.title('Frequency of Each Cancer Stage')
        plt.xlabel('Cancer Stage')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()


    def plot_protein_marker_correlations(self):
        """Plot a heatmap of the correlation matrix for all protein markers with LC and NSCLC values."""
        # Select only the columns related to protein markers and model scores
        data_subset = self.df[self.numerical_vars + self.models]

        corr_matrix = data_subset.corr()

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

    def plot_model_score_vs_model_score(self, model_name):
        """Generate scatter plots overlayed with density plots for each model against nodule size, differentiated by categorical variable using shapes."""
        for model in self.models:
            for var in self.categorical_vars:
                fig, ax = plt.subplots(figsize=(10, 6))

                valid_data = self.df.dropna(subset=[model_name, model, var])

                # KDE plot overlay
                sns.kdeplot(x=model_name, y=model, data=valid_data, fill=True,
                            levels=5, alpha=0.7, color='grey', ax=ax)

                marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']

                value_mapping = {'ja': 'yes', 'nee': 'no'}
                valid_data[var] = valid_data[var].map(value_mapping).fillna(valid_data[var])
                unique_values_mapped = valid_data[var].unique()

                # Ensure markers dictionary uses mapped values
                markers = {value: marker for value, marker in zip(unique_values_mapped, marker_styles)}

                # Scatter plot with mapped category values
                sns.scatterplot(x=model_name, y=model, data=valid_data,
                                style=var, hue=var, markers=markers, edgecolor="none",
                                legend='full', s=100, ax=ax)

                # Calculate Spearman correlation coefficient
                # rho, p_value = spearmanr(valid_data[model_name], valid_data[model])

                # Annotate plot with Spearman rho value
                # plt.annotate(f"Spearman ρ = {rho:.2f} (p = {p_value:.3f})", xy=(0.5, 0.95),
                             # xycoords='axes fraction', ha='center', fontsize=10,
                             # backgroundcolor='white')

                plt.title(f'Scatter and Density Plot of {model} vs. {model_name} by {var}', fontsize=10)
                plt.xlabel(model_name)
                plt.ylabel(f'{model} Score')
                plt.legend(title=var)
                plt.tight_layout()
                plt.show()

    def plot_model_score_vs_model_score_by_diagnosis(self, model_name):
        """Generate scatter plots overlayed with density plots for each model against nodule size, differentiated by diagnosis."""
        for model in self.models:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use valid_data which has dropped NaN values for the current model and 'Diagnose'
            valid_data = self.df.dropna(subset=['Nodule size (1-30 mm)', model, 'Diagnose'])

            # KDE plot overlay
            sns.kdeplot(x=model_name, y=model, data=valid_data, fill=True,
                        levels=5, alpha=0.7, color='grey', ax=ax)

            # Scatter plot with points colored by 'Diagnose'
            sns.scatterplot(x=model_name, y=model, data=valid_data,
                            hue='Diagnose', style='Diagnose',
                            markers={'No LC': 'o', 'NSCLC': 's'},  # Define markers for each diagnosis
                            edgecolor="none", s=100, ax=ax)  # Increased s for better visibility

            # Calculate Spearman correlation coefficient
            # rho, p_value = spearmanr(valid_data['Nodule size (1-30 mm)'], valid_data[model])

            # Annotate the plot with Spearman rho value
            # plt.annotate(f"Spearman ρ = {rho:.2f} (p = {p_value:.3f})", xy=(0.5, 0.95),
                         # xycoords='axes fraction', ha='center', fontsize=10,
                         # backgroundcolor='white')

            plt.title(f'Scatter and Density Plot of {model} vs. {model_name}')
            plt.xlabel(model_name)
            plt.ylabel(f'{model} Score')
            plt.legend(title='Diagnose')
            plt.tight_layout()
            plt.show()

    def plot_roc_curve(self, true_label_col, score_col, n_splits=5, random_state=42):
        y = self.df[true_label_col].values
        scores = self.df[score_col].values

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(10, 8))

        for fold, (train, test) in enumerate(cv.split(scores.reshape(-1, 1), y)):
            fpr, tpr, thresholds = roc_curve(y[test], scores[test])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold} (AUC = {roc_auc:.2f})')

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

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve for {score_col}')
        plt.legend(loc="lower right")
        plt.show()


    def plot_confusion_matrices(self, threshold=50):
        """
        Plots confusion matrices and calculates sensitivity and specificity for each model based on a specified threshold.

        Parameters:
        - threshold: The threshold for predicting the positive class (default is 10% or 0.1).
        """
        for model in self.models:
            clean_df = self.df.dropna(subset=[model, 'Diagnosis_Encoded'])

            # Apply threshold to model scores to generate binary predictions
            predictions = (clean_df[model] >= threshold).astype(int)

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

    def test_numerical_variable_significance_with_diagnosis(self):
        """Test the association between protein markers and binary diagnosis using logistic regression."""
        print("Testing association between protein markers and diagnosis:\n" + "="*60)

        self.df['Diagnosis_Encoded'] = np.where(self.df['Diagnose'] == 'No LC', 0, 1)

        for marker in self.numerical_vars:
            print(f"\nTesting association for marker: {marker}\n" + "="*60)

            # Prepare the data: drop rows with NaN in either the marker or the encoded diagnosis
            data = self.df.dropna(subset=[marker, 'Diagnosis_Encoded'])

            # Logistic regression/Wald Test
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

    def test_numerical_variable_significance_with_model(self):
        """Calculate and print the Pearson correlation coefficient and p-value for each protein marker and model score."""
        print("Testing correlation and significance between protein markers and model scores:\n" + "="*60)
        for model in self.models:
            print(f"\nTesting significance for model: {model}\n" + "="*60)
            for marker in self.numerical_vars:
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

    def calculate_vif_for_numerical_vars(self):
        """
        Calculates Variance Inflation Factors (VIF) for each protein marker in the DataFrame.
        Returns:
        - A pandas DataFrame containing the VIF for each protein marker.
        """
        # Ensure only valid protein markers present in the DataFrame are considered
        valid_numerical_vars = [marker for marker in self.numerical_vars if marker in self.df.columns]

        X = add_constant(self.df[valid_numerical_vars])

        vif_data = pd.DataFrame()
        vif_data["Protein Marker"] = X.columns.drop('const')  # Exclude the constant term from the results
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(1, len(X.columns))]  # Skip the first index for 'const'

        return vif_data

filepath = 'Dataset BEP Rafi.xlsx'  # Update with your actual file path
eda = ModelEDA(filepath)
#eda.display_dataset_info()
# eda.display_summary_statistics()
#eda.plot_distributions_model_score()
#eda.plot_categorical_frequency_with_diagnosis()
#eda.plot_relationship_with_stadium()
#eda.plot_stadium_frequency()
# eda.plot_node_size_distributions()
# eda.plot_numerical_vars_distribution()
# eda.plot_model_score_vs_model_score('Herder score (%)')
# eda.plot_model_score_vs_model_score_by_diagnosis('Herder score (%)')
# eda.generate_summary_table()
# eda.plot_confusion_matrices()
# eda.calculate_best_sensitivity_specificity()
# eda.test_significance_of_categorical_variables_with_model()
# eda.test_protein_marker_significance_with_model()
# eda.test_significance_of_categorical_variables_with_diagnosis()
# eda.test_numerical_variable_significance_with_diagnosis()
vif_results=eda.calculate_vif_for_numerical_vars()
print(vif_results)

#=========================#
# SIMPLE MODEL EVALUATION #
#=========================#

# Preprocess the data
# eda.preprocess_data()
# model_score = 'Herder score (%)' # Change to the specified model score
# target_variable = 'Diagnosis_Encoded'
# threshold = 10
# avg_metrics = eda.evaluate_simple_model_scores(target_variable, model_score, threshold)
# print(avg_metrics)
# eda.plot_prediction_histogram(target_variable, model_score, threshold)
# eda.plot_roc_curve(target_variable, model_score)
