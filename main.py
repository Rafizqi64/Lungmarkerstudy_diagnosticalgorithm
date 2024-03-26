import numpy as np
import pandas as pd

from data_preprocessing import DataPreprocessor
from ensemble_model import VotingModel, score_based_ensemble
from model import Model

df = pd.read_excel('Dataset BEP Rafi.xlsx')
filepath='Dataset BEP Rafi.xlsx'
target='Diagnose'
binary_map={'Nee': 0, 'Ja': 1}
preprocessor = DataPreprocessor(filepath, target, binary_map)

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
            'remainder__Current/Former smoker',
            'remainder__Previous History of Extra-thoracic Cancer',
            'remainder__Nodule size (1-30 mm)',
            'remainder__Nodule Upper Lobe',
            'remainder__Spiculation',
        ]

features_herbert = [
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

features_ensemble_output = [
            'remainder__Brock score (%)',
            'remainder__Herder score (%)',
            'remainder__% LC in TM-model',
            'remainder__% NSCLC in TM-model'
        ]

features_BH_output = [
            'remainder__Brock score (%)',
            'remainder__Herder score (%)',
        ]

features_LBx_output = [
            'remainder__% LC in TM-model',
            'remainder__% NSCLC in TM-model'
        ]

# Initialize the Model class
model_manager = Model(filepath, target, binary_map, threshold_metric='npv')


# features_ensemble = list(set(features_brock + features_herder + features_lbx))
features_brock_and_herder = list(set(features_brock + features_herbert))

# Clear previously stored models and their results
model_manager.reset_models()

# Add models to the manager
model_manager.add_model("brock", features_brock)
# model_manager.add_model("herder", features_herder)
model_manager.add_model("herbert", features_herbert)
model_manager.add_model("lbx", features_lbx)
# model_manager.add_model("lc", features_lc)
# model_manager.add_model("nsclc", features_nsclc)

# Feature selection
# model_manager.apply_tree_based_feature_selection("brock")
# model_manager.apply_tree_based_feature_selection("herder")
# model_manager.apply_tree_based_feature_selection("herbert")
model_manager.apply_tree_based_feature_selection("lbx")

model_manager.apply_rfe_feature_selection("brock")
# model_manager.apply_rfe_feature_selection("herder")
model_manager.apply_rfe_feature_selection("herbert")
# model_manager.apply_rfe_feature_selection("lbx")

# model_manager.apply_logistic_l1_feature_selection("brock")
# model_manager.apply_logistic_l1_feature_selection("herder")
# model_manager.apply_logistic_l1_feature_selection("herbert")
# model_manager.apply_logistic_l1_feature_selection("lbx")

# Train models and prepare the voting ensemble
features_pmBHscore = features_brock_and_herder + features_LBx_output
model_manager.add_model("Lbx scores + Brock and herder features", features_pmBHscore)
features_ensemble = model_manager.get_updated_ensemble_features()

# features_ensembleLogistic = features_ensemble
# model_manager.add_model("ensembleLogistic", features_ensembleLogistic)

trained_models = model_manager.train_models()


#===========================#
#     INDIVIDUAL MODELS     #
#===========================#


# print("\nBrock Formula")
# print(model_manager.get_logistic_regression_formula('brock'))
# model_manager.plot_roc_curves('brock')
# model_manager.generate_shap_plot('brock')
# model_manager.plot_prediction_histograms('brock')


# model_manager.plot_roc_curves('herder')
# model_manager.generate_shap_plot('herder')
# model_manager.plot_prediction_histograms('herder')


# print("\nHerbert Formula")
# print(model_manager.get_logistic_regression_formula('herbert'))
# model_manager.plot_roc_curves('herbert')
# model_manager.generate_shap_plot('herbert')
# model_manager.plot_prediction_histograms('herbert')

# print("\nlc Formula")
# print(model_manager.get_logistic_regression_formula('lc'))
# model_manager.plot_roc_curves('lc')
# model_manager.generate_shap_plot('lc')
# model_manager.plot_prediction_histograms('lc')

# print("\nnsclc Formula")
# print(model_manager.get_logistic_regression_formula('nsclc'))
# model_manager.plot_roc_curves('nsclc')
# model_manager.generate_shap_plot('nsclc')
# model_manager.plot_prediction_histograms('nsclc')

# print("\nlbx Formula")
# print(model_manager.get_logistic_regression_formula('lbx'))
# model_manager.plot_roc_curves('lbx')
# model_manager.generate_shap_plot('lbx')
# model_manager.plot_prediction_histograms('lbx')


#===========================#
#       FULL ENSEMBLE       #
#===========================#


print("\nLbx scores + Brock and herder features Formula")
print(model_manager.get_logistic_regression_formula("Lbx scores + Brock and herder features"))
# model_manager.plot_roc_curves("Lbx scores + Brock and herder features")
# model_manager.generate_shap_plot("Lbx scores + Brock and herder features")
model_manager.plot_prediction_histograms("Lbx scores + Brock and herder features")
model_manager.plot_confusion_matrices("Lbx scores + Brock and herder features")

# print("\nensembleLogistic Formula")
# print(model_manager.get_logistic_regression_formula('ensembleLogistic'))
# model_manager.plot_roc_curves('ensembleLogistic')
# model_manager.generate_shap_plot('ensembleLogistic', features_brock)
# model_manager.plot_prediction_histograms('ensembleLogistic')


voting_model = VotingModel(trained_models, features_ensemble, filepath, target, binary_map, 'VOTING INPUT', threshold_metric='npv')
voting_model.reset()
voting_model.train_voting_classifier()
voting_model.plot_roc_curves()
# voting_model.generate_shap_plot()
# voting_model.plot_prediction_histograms()
voting_model.plot_confusion_matrices()


score_model = score_based_ensemble(filepath, target, binary_map, features_ensemble_output, "ENSEMBLE OUTPUT", threshold_metric='npv')
score_model.fit_evaluate()
score_model.print_scores()
score_model.plot_roc_curve()
score_model.plot_prediction_histogram()
score_model.plot_confusion_matrices()

#===========================#
# BROCK AND HERDER ENSEMBLE #
#===========================#

# voting_model = VotingModel(trained_models, features_brock_and_herder, filepath, target, binary_map, 'BROCK AND HERBERT INPUT', threshold_metric='npv')
# voting_model.reset()
# voting_model.train_voting_classifier()
# voting_model.plot_roc_curves()
# voting_model.generate_shap_plot()
# voting_model.plot_prediction_histograms()
# voting_model.plot_confusion_matrices()

# score_model = score_based_ensemble(filepath, target, binary_map, features_BH_output, "BROCK AND HERDER OUTPUT", threshold_metric='npv')
# score_model.fit_evaluate()
# score_model.print_scores()
# score_model.plot_roc_curve()
# score_model.plot_prediction_histogram()
# score_model.plot_confusion_matrices()
