import numpy as np
import pandas as pd

from brock_and_herder import BrockAndHerderModel
from data_preprocessing import DataPreprocessor
from ensemble_model import ensemble_model
from lbx_model import LBxModel

df = pd.read_excel('Dataset BEP Rafi.xlsx')
# Define features for each model
# Initialize Brock and Herder models directly with their respective features and the dataset
brock_herder_model = BrockAndHerderModel(filepath='Dataset BEP Rafi.xlsx', target='Diagnose', binary_map={'Nee': 0, 'Ja': 1})
brock_herder_model.train_models()

model_brock, model_herder = brock_herder_model.get_models()

lbx_model = LBxModel(filepath='Dataset BEP Rafi.xlsx', target='Diagnose', binary_map={'Nee': 0, 'Ja': 1})
lbx_model.train_models()

# Retrieve the trained models
model_lc, model_nsclc = lbx_model.get_models()


preprocessor = DataPreprocessor(filepath='Dataset BEP Rafi.xlsx', target='Diagnose', binary_map={'Nee': 0, 'Ja': 1})

# Load and preprocess the data
X, y = preprocessor.load_and_transform_data()


# Create and use the Ensemble Model
models = [
    ('brock', model_brock),
    ('herder', model_herder),
    ('lbx1', model_lc),
    ('lbx2', model_nsclc)
]
evaluator = ensemble_model(models=models)
evaluator.fit_evaluate(X, y)
evaluator.print_scores()

