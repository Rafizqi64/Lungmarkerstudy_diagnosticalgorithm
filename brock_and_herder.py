import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Assuming 'df' is your DataFrame
df = pd.read_excel('Dataset BEP Rafi.xlsx')

# Map binary features to 0 and 1
binary_map = {'Nee': 0, 'Ja': 1}
df.replace(binary_map, inplace=True)

# Define the target variable
target = 'Diagnose'
df[target] = df[target].apply(lambda x: 0 if x == 'No LC' else 1)
brock_features = [
    'Family History of LC', 'Current/Former smoker', 'Emphysema',
    'Nodule size (1-30 mm)', 'Nodule Upper Lobe', 'Nodule Count',
    'Nodule Type']

herder_features = [
    'Current/Former smoker', 'Previous History of Extra-thoracic Cancer',
    'Nodule size (1-30 mm)', 'Nodule Upper Lobe', 'Spiculation',
    'PET-CT Findings']


def train_evaluate_model(df, features, target, model_name):
    y = df[target].values
    X = df[features]

    # Define categorical features for one-hot encoding
    categorical_features = [f for f in features if f not in binary_map.keys()]
    # Preprocessing pipeline adjustment
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    # Define the model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(solver='liblinear'))
    ])

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': [], 'f1': []}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]

        # Update scores
        scores['accuracy'].append(accuracy_score(y_test, predictions))
        scores['precision'].append(precision_score(y_test, predictions))
        scores['recall'].append(recall_score(y_test, predictions))
        scores['f1'].append(f1_score(y_test, predictions))
        scores['roc_auc'].append(roc_auc_score(y_test, proba))

    # After the last fold
    model = pipeline.named_steps['model']
    final_features = categorical_features + [f for f in features if f in binary_map.keys()]
    encoded_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(final_features)
    all_features = np.concatenate([encoded_features, [f for f in features if f in binary_map.keys()]])

    # Constructing the formula
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    terms = [f"{coeff:.4f}*{feature}" for coeff, feature in zip(coefficients, all_features)]
    formula = f"logit(p) = {intercept:.4f} + " + " + ".join(terms)
    print(f"\n{model_name} Model Logistic Regression Formula:\n{formula}\n")

    # Calculate and print average scores
    for metric, score_list in scores.items():
        print(f"{metric.capitalize()} (average): {np.mean(score_list):.4f}")

# Train and evaluate Brock model
print("Brock Model Scores:")
train_evaluate_model(df, brock_features, target, "Brock")

# Train and evaluate Herder model
print("\nHerder Model Scores:")
train_evaluate_model(df, herder_features, target, "Herder")

