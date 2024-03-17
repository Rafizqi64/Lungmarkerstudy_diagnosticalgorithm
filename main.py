import pandas as pd

from data_preprocessing import DataPreprocessor
from ensemble_model import VotingModel, score_based_ensemble
from model import Model

df = pd.read_excel('Dataset BEP Rafi.xlsx')
filepath='Dataset BEP Rafi.xlsx'
target='Diagnose'
binary_map={'Nee': 0, 'Ja': 1}
preprocessor = DataPreprocessor(filepath, target, binary_map)

# Initialize the Model class
model_manager = Model(filepath, target, binary_map)

# Define the feature sets for each model
features_brock = [
            'cat__Nodule Type_GroundGlass',
            'cat__Nodule Type_PartSolid',
            'remainder__Family History of LC',
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
            'remainder__Current/Former smoker',
            'remainder__Previous History of Extra-thoracic Cancer',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Spiculation',
        ]


features_lc = [
            'remainder__CEA',
            'remainder__CYFRA 21-1',
         ]

features_nsclc = [
            'remainder__CEA',
            'remainder__CYFRA 21-1',
            'remainder__proGRP',
            'remainder__NSE corrected for H-index',
        ]

features_lbx = [
            'remainder__CEA',
            'remainder__CYFRA 21-1',
            'remainder__CA15.3',
            'remainder__CA125',
            'remainder__proGRP',
            'remainder__NSE corrected for H-index',
            'remainder__HE4',
            'remainder__SCCA'
        ]

features_score = [
            'remainder__Brock score (%)',
            'remainder__Herder score (%)',
            'remainder__% LC in TM-model',
            'remainder__% NSCLC in TM-model'
        ]

features_BH_output = [
            'remainder__Brock score (%)',
            'remainder__Herder score (%)',
        ]

features_ensemble = list(set(features_brock + features_herder + features_lbx))
features_brock_and_herder = list(set(features_brock + features_herder))

# Clear previously stored models and their results
model_manager.reset_models()

# Add models to the manager
model_manager.add_model("brock", features_brock)
model_manager.add_model("herder", features_herder)
model_manager.add_model("lbx", features_lbx)
# model_manager.add_model("lc", features_lc)
# model_manager.add_model("nsclc", features_nsclc)

# Train models and prepare the voting ensemble
model_manager.apply_rfe_feature_selection("lbx")
trained_models = model_manager.train_models()

# print("\nBrock Formula")
# print(model_manager.get_logistic_regression_formula('brock'))
# model_manager.plot_roc_curves('brock')
# model_manager.generate_shap_plot('brock', features_brock)
# model_manager.plot_prediction_histograms('brock')

# print("\nlc Formula")
# print(model_manager.get_logistic_regression_formula('lc'))
# model_manager.plot_roc_curves('lc')
# model_manager.generate_shap_plot('lc', features_lc)
# model_manager.plot_prediction_histograms('lc')

# print("\nnsclc Formula")
# print(model_manager.get_logistic_regression_formula('nsclc'))
# model_manager.plot_roc_curves('nsclc')
# model_manager.generate_shap_plot('nsclc', features_nsclc)
# model_manager.plot_prediction_histograms('nsclc')

# print("\nlbx Formula")
# print(model_manager.get_logistic_regression_formula('lbx'))
# model_manager.plot_roc_curves('lbx')
# model_manager.generate_shap_plot('lbx', features_lbx)
# model_manager.plot_prediction_histograms('lbx')


#===========================#
#       FULL ENSEMBLE       #
#===========================#

voting_model = VotingModel(trained_models, features_ensemble, filepath, target, binary_map, 'ensemble')
voting_model.reset()
voting_model.train_voting_classifier()
# voting_model.plot_roc_curves()
# voting_model.generate_shap_plot()
# voting_model.plot_prediction_histograms()

score_model = score_based_ensemble(filepath, target, binary_map, features_score, "score based ensemble")
score_model.fit_evaluate()
score_model.print_scores()
# score_model.plot_roc_curve()
# score_model.plot_prediction_histogram()
# Reset any previous state in the voting model and train the voting classifier


#===========================#
# BROCK AND HERDER ENSEMBLE #
#===========================#

voting_model = VotingModel(trained_models, features_brock_and_herder, filepath, target, binary_map, 'Brock and Herder input')
voting_model.reset()
voting_model.train_voting_classifier()
# voting_model.plot_roc_curves()
# voting_model.generate_shap_plot()
# voting_model.plot_prediction_histograms()

score_model = score_based_ensemble(filepath, target, binary_map, features_BH_output, "Brock and Herder output")
score_model.fit_evaluate()
score_model.print_scores()
# score_model.plot_roc_curve()
# score_model.plot_prediction_histogram()


