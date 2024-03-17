import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, filepath, target, binary_map):
        self.filepath = filepath
        self.target = target
        self.binary_map = binary_map
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = []

    def load_and_transform_data(self):
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

        # Apply Brock model transformations for Nodule size and count
        if 'Nodule size' in self.df.columns:
            self.df['Nodule size'] = (self.df['Nodule size'] / 10) - 0.5 - 1.58113883
        if 'Nodule count' in self.df.columns:
            self.df['Nodule count'] = self.df['Nodule count'] - 4

        # preprocess dataset
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        # separate features and target
        X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

        # Apply preprocessing and convert to DataFrame for feature names retention
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Convert the processed array back to a DataFrame with new feature names
        self.X = pd.DataFrame(X_transformed, columns=self.feature_names)
        return self.X, self.y
