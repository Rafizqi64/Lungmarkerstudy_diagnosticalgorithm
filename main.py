import pandas as pd

from brock_and_herder import BrockAndHerderModel
from ensemble_model import ensemble_model, score_based_ensemble
from lbx_model import LBxModel

df = pd.read_excel('Dataset BEP Rafi.xlsx')
filepath='Dataset BEP Rafi.xlsx'
target='Diagnose'
binary_map={'Nee': 0, 'Ja': 1}
# Define features for each model
# Initialize Brock and Herder models directly with their respective features and the dataset
print(10*'='+"Brock and Herder"+10*"=")
brock_herder_model = BrockAndHerderModel(filepath, target, binary_map)
brock_herder_model.train_models()
model_brock, model_herder = brock_herder_model.get_models()

print(10*'='+"LC and NSCLC"+10*"=")
lbx_model = LBxModel(filepath, target, binary_map)
lbx_model.train_models()

# Retrieve the trained models
model_lc, model_nsclc = lbx_model.get_models()

# Create and use the Ensemble Model
models = [
    ('brock', model_brock),
    ('herder', model_herder),
    ('lbx1', model_lc),
    ('lbx2', model_nsclc)
]

print(10*'='+"Score Based Ensemble"+10*"=")
scored_evaluator = score_based_ensemble(filepath, target, binary_map)
scored_evaluator.fit_evaluate()
scored_evaluator.print_scores()


print(10*'='+"Voting Classifier Ensemble"+10*"=")
evaluator = ensemble_model(models, filepath, target, binary_map)
evaluator.fit_evaluate()
evaluator.print_scores()


#Plot probalilities
lbx_model.plot_prediction_histograms()
brock_herder_model.plot_prediction_histograms()
evaluator.plot_prediction_histogram()
scored_evaluator.plot_prediction_histogram()

#Plot Mean ROC Curve
lbx_model.plot_roc_curves(lbx_model.lc_results, 'LC')
lbx_model.plot_roc_curves(lbx_model.nsclc_results, 'NSCLC')
brock_herder_model.plot_roc_curves(brock_herder_model.brock_results, 'Brock')
brock_herder_model.plot_roc_curves(brock_herder_model.herder_results, 'Herder')
evaluator.plot_roc_curve()
scored_evaluator.plot_roc_curve()
