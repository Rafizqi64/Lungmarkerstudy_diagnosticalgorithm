import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LungCancerEDA:
    def __init__(self, filepath):
        """Initialize the EDA object and load the dataset."""
        self.df = pd.read_excel(filepath)
        self.numerical_features = ['Nodule size (1-30 mm)', 'Brock score', 'Herder score (%)', 'CA125', 'CA15.3', 'CEA', 'CYFRA 21-1', 'HE4', 'NSE', 'NSE corrected for H-index', 'proGRP', 'SCCA', '% LC in TM-model', '% NSCLC in TM-model']
        self.categorical_features = ['Diagnose', 'Stadium', 'Family History of LC', 'Current/Former smoker', 'Previous History of Extra-thoracic Cancer', 'Emphysema', 'Nodule Type', 'Nodule Upper Lobe', 'Nodule Count', 'Spiculation', 'PET-CT Findings']

    def overview(self):
        """Print an overview of the dataset including info and summary statistics."""
        print(self.df.head())
        self.df.info()
        print(self.df.describe())

    def plot_numerical_distributions(self):
        """Visualize distributions of numerical features."""
        for feature in self.numerical_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[feature], kde=True, bins=20)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.show()

    def plot_categorical_distributions(self):
        """Visualize distributions of categorical features."""
        for feature in self.categorical_features:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=self.df[feature], order = self.df[feature].value_counts().index)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Count')
            plt.ylabel(feature)
            plt.show()

    def correlation_matrix(self):
        """Generate and plot the correlation matrix for numerical features."""
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.df[self.numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

    def multivariate_analysis(self, features, hue):
        """Conduct and visualize multivariate analysis."""
        sns.pairplot(self.df, vars=features, hue=hue)
        plt.show()

# Usage
eda = LungCancerEDA('Dataset BEP Rafi.xlsx')
eda.overview()  # Basic dataset overview
eda.plot_numerical_distributions()  # Distribution of numerical features
eda.plot_categorical_distributions()  # Distribution of categorical features
eda.correlation_matrix()  # Correlation matrix

# For multivariate analysis, specify features and hue
features = ['', 'Brock score', 'Herder score (%)']
eda.multivariate_analysis(features, 'Diagnose')

