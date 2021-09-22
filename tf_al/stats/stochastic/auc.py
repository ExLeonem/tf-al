from scipy.special import comb
from sklearn.metrics import roc_auc_score
import numpy as np

class AUC:

    def __init__(self, multi_target=False):
        self.name = "stochastic_auc"

    
    def __call__(self, true_target, pred_target):
        pass