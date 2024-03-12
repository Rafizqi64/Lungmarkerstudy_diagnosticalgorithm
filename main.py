import pandas as pd

from ensemble_model import ensemble_model, score_based_ensemble
# from brock_and_herder import BrockAndHerderModel
from model import Model

# from lbx_model import LBxModel

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


# Add models to the manager
model_manager.add_model("brock", features_brock)
model_manager.add_model("herder", features_herder)
model_manager.add_model("lc", features_lc)
model_manager.add_model("nsclc", features_nsclc)

# Assuming 'trainer' is an instance of the class containing the 'train_models' method
trained_models = model_manager.train_models()

# ROC Curve# s
# model_manager.plot_roc_curves('brock')
# model_manager.plot_roc_curves('herder')
# model_manager.plot_roc_curves('lc')
# model_manager.plot_roc_curves('nsclc')

# Learning Curves

model_manager.generate_learning_curve('brock')
model_manager.generate_learning_curve('herder')
model_manager.generate_learning_curve('lc')
model_manager.generate_learning_curve('nsclc')

# SHAP Plots
# model_manager.generate_shap_plot('brock', features_brock)
# model_manager.generate_shap_plot('herder', features_herder)
# model_manager.generate_shap_plot('lc', features_lc)
# model_manager.generate_shap_plot('nsclc', features_nsclc)

# Prediction histograms
# model_manager.plot_prediction_histograms('brock')
# model_manager.plot_prediction_histograms('herder')
# model_manager.plot_prediction_histograms('lc')
# model_manager.plot_prediction_histograms('nsclc')

# # Extract models for the ensemble
# models = [
    # ('brock', trained_models['brock']),
    # ('herder', trained_models['herder']),
    # ('nsclc', trained_models['nsclc']),
# ]


# print(10*'='+"Score Based Ensemble"+10*"=")
# scored_evaluator = score_based_ensemble(filepath, target, binary_map)
# scored_evaluator.fit_evaluate()
# scored_evaluator.print_scores()

# print(10*'='+"Voting Classifier Ensemble"+10*"=")
# evaluator = ensemble_model(models, filepath, target, binary_map)
# evaluator.fit_best_model()

# # #Plot learning curves
# # #Plot probalilities
# # evaluator.plot_prediction_histograms()
# scored_evaluator.plot_prediction_histogram()

# #Plot Mean ROC Curve
# lbx_model.plot_roc_curves(lbx_model.lc_results, 'LC')
# lbx_model.plot_roc_curves(lbx_model.nsclc_results, 'NSCLC')
# brock_herder_model.plot_roc_curves(brock_herder_model.brock_results, 'Brock')
# brock_herder_model.plot_roc_curves(brock_herder_model.herder_results, 'Herder')
# evaluator.plot_roc_curves(ensemble_model.ensemble_results)
# scored_evaluator.plot_roc_curve()
