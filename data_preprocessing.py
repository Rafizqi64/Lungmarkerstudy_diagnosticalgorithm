import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    def __init__(self, filepath, target, binary_map):
        self.filepath = filepath
        self.target = target
        self.binary_map = binary_map
        self.df = None
        self.X = None
        self.y = None

    def load_and_transform_data(self):
        # Load data
        self.df = pd.read_excel(self.filepath)
        # Exclude specified columns
        columns_to_exclude = ['IDNR', 'Stadium', 'Brock score (%)', 'Herder score (%)', 'CA125', 'CA15.3', 'HE4', 'NSE corrected for H-index', 'SCCA', 'ctDNA mutatie', '% LC in TM-model', '% NSCLC in TM-model']
        self.df.drop(columns_to_exclude, axis=1, inplace=True)

        # Apply binary mapping
        self.df.replace(self.binary_map, inplace=True)

        # Convert target variable
        self.df[self.target] = self.df[self.target].apply(lambda x: 0 if x == 'No LC' else 1)

        # Identify categorical features for one-hot encoding
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        # Apply log transformation to numerical features likely to be skewed
        numerical_features = ['CYFRA 21-1', 'CEA', 'NSE', 'proGRP']
        for column in numerical_features:
            if column in self.df.columns:
                self.df[column] = np.log10(self.df[column] + 1)  # +1 to avoid log(0)

        # preprocess dataset
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        # separate features and target
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

        # Apply preprocessing
        self.X = self.preprocessor.fit_transform(self.X)

        return self.X, self.y

    def get_feature_indices(self, features):
        # Get names of transformed features
        transformed_features = self.preprocessor.get_feature_names_out()
        # Initialize a list to hold indices for the specified features
        feature_indices_flat = []

        # Iterate over each specified feature
        for feature in features:
            if feature in self.df.select_dtypes(include=['object']).columns:
                # Feature is categorical and has been one-hot encoded
                encoded_features_indices = [i for i, f in enumerate(transformed_features) if f.startswith(f"cat__{feature}")]
                feature_indices_flat.extend(encoded_features_indices)
            else:
                # For numerical or binary features, find their transformed index
                # Assume numerical/binary features are tagged with 'remainder__' in transformed features
                for transformed_feature in transformed_features:
                    if transformed_feature.endswith(f'__{feature}'):
                        index = list(transformed_features).index(transformed_feature)
                        feature_indices_flat.append(index)
                        break  # Assume each feature is unique and stop after finding the first match
        return feature_indices_flat
