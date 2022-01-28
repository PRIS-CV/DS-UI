import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score

#### Gaussian Mixture Model Uncertainty for Classification
# probs: narray, [batch size, #class], softmax(outputs of the gmm layer)
# alphas: narray, [batch size, #class], exp(outputs of the gmm layer)
def gmm_max_probability(probs, epsilon=1e-8):
    max_pred_prob = np.max(probs, axis=1)
    return max_pred_prob

def gmm_entropy_expected(probs, epsilon=1e-8):
    probs = np.asarray(probs, dtype=np.float64) + epsilon
    entropy_expected = np.asarray([entropy(prob) for prob in probs], dtype=np.float64)
    return entropy_expected

def gmm_uncertainty(probs, alphas, epsilon=1e-8):
    entropy_expected = gmm_entropy_expected(probs)
    max_prob = gmm_max_probability(probs)
    return max_prob, entropy_expected

def gmm_auc(probs, alphas, targets):
    max_prob, ent = gmm_uncertainty(probs, alphas)
    predictions = np.argmax(probs, axis=1)
    rightwrong = np.asarray(targets != predictions, dtype=np.int32)
    print(np.sum(rightwrong))
    return (roc_auc_score(rightwrong, 1. - max_prob), roc_auc_score(rightwrong, ent), 
           average_precision_score(rightwrong, 1. - max_prob), average_precision_score(rightwrong, ent))
