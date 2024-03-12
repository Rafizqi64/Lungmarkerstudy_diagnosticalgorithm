import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ensemble_model import VotingModel, score_based_ensemble
from model import Model

df = pd.read_excel('Dataset BEP Rafi.xlsx')
filepath='Dataset BEP Rafi.xlsx'
target='Diagnose'
binary_map={'Nee': 0, 'Ja': 1}
# Initialize the Model class
model_manager = Model(filepath, target, binary_map)

# Define the feature sets for each model
features_brock = [
            'cat__Nodule Type_GroundGlass',
            'cat__Nodule Type_PartSolid',
            'cat__Nodule Type_Solid',
            'remainder__Family History of LC',
            'remainder__Current/Former smoker',
            'remainder__Emphysema',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Nodule Count',
            'remainder__Spiculation',
        ]

features_herder = [
            'cat__PET-CT Findings_Faint',
            'cat__PET-CT Findings_Intense',
            'cat__PET-CT Findings_Moderate',
            'cat__PET-CT Findings_No FDG avidity',
            'remainder__Family History of LC',
            'remainder__Previous History of Extra-thoracic Cancer',
            'remainder__Emphysema',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Nodule Count',
            'remainder__Spiculation',
        ]


features_lc = [
            'remainder__CEA',
            'remainder__CYFRA 21-1',
        ]


features_nsclc = [
            'remainder__CEA',
            'remainder__CYFRA 21-1',
            'remainder__NSE',
            'remainder__proGRP'
        ]
features_ensemble = list(set(features_brock + features_herder + features_nsclc + features_lc))

# Clear previously stored models and their results
model_manager.reset_models()

# Add models to the manage# r
model_manager.add_model("brock", features_brock)
model_manager.add_model("herder", features_herder)
model_manager.add_model("lc", features_lc)
model_manager.add_model("nsclc", features_nsclc)

# Train models and prepare the voting ensemble
trained_models = model_manager.train_models()
voting_model = VotingModel(trained_models, features_ensemble, filepath, target, binary_map)

score_model = score_based_ensemble(filepath, target, binary_map)
score_model.fit_evaluate()
score_model.print_scores()

# Reset any previous state in the voting model and train the voting classifier
voting_model.reset()
voting_model.train_voting_classifier()

model_manager.plot_roc_curves('brock')
model_manager.generate_shap_plot('brock', features_brock)
model_manager.plot_prediction_histograms('brock')

model_manager.plot_roc_curves('herder')
model_manager.generate_shap_plot('herder', features_herder)
model_manager.plot_prediction_histograms('herder')

model_manager.plot_roc_curves('lc')
model_manager.generate_shap_plot('lc', features_lc)
model_manager.plot_prediction_histograms('lc')

model_manager.plot_roc_curves('nsclc')
model_manager.generate_shap_plot('nsclc', features_nsclc)
model_manager.plot_prediction_histograms('nsclc')

voting_model.plot_roc_curves()
voting_model.generate_shap_plot()
voting_model.plot_prediction_histograms()

score_model.plot_roc_curve()
score_model.plot_prediction_histogram()
