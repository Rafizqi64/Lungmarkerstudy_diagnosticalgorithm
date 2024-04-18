import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """
    Prepares and transforms dataset for model training and evaluation.

    Attributes:
    - filepath (str): The path to the dataset file.
    - target (str): The name of the target variable in the dataset.
    - binary_map (dict): A dictionary mapping original target labels to binary values.
    - df (DataFrame): The loaded dataset after initial processing.
    - X (DataFrame): The feature matrix after applying all transformations.
    - y (Series): The Diagnosis after binary mapping.
    - feature_names (list): List of feature names after processing and encoding.

    Methods:
    - load_and_transform_data: Loads data from the specified file path and applies transformations such as
      binary mapping, log transformations for numerical features, standard scaling, and one-hot encoding for categorical features.
      Specific transformations for model types like 'brock' are conditionally applied.

    The class facilitates flexible preprocessing tailored to different modeling scenarios, ensuring data is appropriately
    formatted and ready for use in machine learning models.
    """
    def __init__(self, filepath, target, binary_map):
        self.filepath = filepath
        self.target = target
        self.binary_map = binary_map
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = []

    def load_and_transform_data(self, model_name=None):
        self.df = pd.read_excel(self.filepath)
        # Exclude specified columns
        columns_to_exclude = ['IDNR', 'Stadium', 'ctDNA mutatie']
        self.df.drop(columns_to_exclude, axis=1, inplace=True)

        # Apply binary mapping
        self.df.replace(self.binary_map, inplace=True)

        # Convert target variable
        self.df[self.target] = self.df[self.target].apply(lambda x: 0 if x == 'No LC' else 1)

        # Identify categorical features for one-hot encoding
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        # Apply log transformation to numerical features likely to be skewed
        numerical_features = ['CYFRA 21-1', 'CEA', 'NSE', 'proGRP', 'CA125', 'CA15.3', 'HE4', 'NSE corrected for H-index', 'SCCA']

        for column in numerical_features:
            if column in self.df.columns:
                self.df[column] = np.log10(self.df[column] + 1)  # +1 to avoid log(0)

        # Initialize the StandardScaler
        scaler = StandardScaler()
        # Apply standard scaling to the numerical features
        self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])

        # Apply Brock model transformations for Nodule size and count, if model_name is 'brock'
        if model_name == 'brock':
            if 'Nodule size' in self.df.columns:
                self.df['Nodule size'] = (self.df['Nodule size'] / 10) - 0.5 - 1.58113883
            if 'Nodule count' in self.df.columns:
                self.df['Nodule count'] = self.df['Nodule count'] - 4

        # Preprocess dataset
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        # Separate features and target
        X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

        # Apply preprocessing and convert to DataFrame for feature names retention
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Convert the processed array back to a DataFrame with new feature names
        self.X = pd.DataFrame(X_transformed, columns=self.feature_names)
        return self.X, self.y
