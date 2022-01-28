import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score
from scipy.special import digamma
from scipy.stats import dirichlet, entropy

def aupr(rightwrong, measure, pos_label=1):
    measure = np.asarray(measure, dtype=np.float128)[:, np.newaxis]
    min_measure = np.min(measure)
    if min_measure < 0.0: 
        measure += abs(min_measure)
    measure = np.log(measure + 1e-8)
    if pos_label != 1: 
        measure *= -1.0

    precision, recall, thresholds = precision_recall_curve(rightwrong, measure)
    return auc(recall, precision)

def auroc(rightwrong, measure, pos_label=1):
    measure = np.asarray(measure, dtype=np.float128)[:, np.newaxis]
    min_measure = np.min(measure)
    if min_measure < 0.0: 
        measure += abs(min_measure)
    measure = np.log(measure + 1e-8)
    if pos_label != 1: 
        measure *= -1.0

    fpr, tpr, thresholds = roc_curve(rightwrong, measure)
    return auc(fpr, tpr)

#### Conventional Deep Neural Network Uncertainty for Classification
# probs: narray, [#sample, #class]
def dnn_max_probability(probs, epsilon=1e-8):
    return np.max(probs, axis=1)

def dnn_entropy_expected(probs, epsilon=1e-8):
    return -np.squeeze(np.sum(probs * np.log(probs + epsilon), axis=1))

def dnn_auc(probs, targets):
    max_prob = dnn_max_probability(probs)
    ent = dnn_entropy_expected(probs)
    predictions = np.argmax(probs, axis=1)
    rightwrong = np.asarray(targets != predictions, dtype=np.int32)
    return (roc_auc_score(rightwrong, 1. - max_prob), roc_auc_score(rightwrong, ent), 
           aupr(rightwrong, 1. - max_prob), aupr(rightwrong, ent))


#### Monte-Carlo Dropout Bayesian Uncertainty for Classification
# probs: narray, [#sample, #mc, #class]
def mcdp_expected_entropy(probs, epsilon=1e-8):
    log_probs = np.log(probs + epsilon)
    expected_entropy = -np.mean(np.sum(probs * log_probs, axis=2), axis=1)
    return expected_entropy

def mcdp_entropy_expected(probs, epsilon=1e-8):
    mean_probs = np.mean(probs, axis=1)
    log_mean_probs = np.log(mean_probs + epsilon)
    entropy_expected = -np.squeeze(np.sum(mean_probs * log_mean_probs, axis=1))
    return entropy_expected

def mcdp_mutual_information(probs, epsilon=1e-8):
    entropy_of_expected = mcdp_entropy_expected(probs, epsilon=epsilon)
    expected_entropy = mcdp_expected_entropy(probs, epsilon=epsilon)
    mutual_information = entropy_of_expected - expected_entropy
    return mutual_information

def mcdp_max_probability(probs, epsilon=1e-8):
    mean_probs = np.mean(probs, axis=1)
    max_pred_prob = np.max(mean_probs, axis=1)
    return max_pred_prob

def mcdp_uncertainty(probs):
    max_prob = mcdp_max_probability(probs)
    ent = mcdp_entropy_expected(probs)
    mi = mcdp_mutual_information(probs)
    return max_prob, ent, mi

def mcdp_auc(probs, targets):
    max_prob, ent, mi = mcdp_uncertainty(probs)
    predictions = np.argmax(np.mean(probs, axis=1), axis=1)
    rightwrong = np.asarray(targets != predictions, dtype=np.int32)
    return (roc_auc_score(rightwrong, 1. - max_prob), roc_auc_score(rightwrong, ent), roc_auc_score(rightwrong, mi), 
           aupr(rightwrong, 1. - max_prob), aupr(rightwrong, ent), aupr(rightwrong, mi))

#### Dirichlet Prior Network Uncertainty for Classification
# alphas: narray, [#sample, #class], exp(outputs of the last fc layer)
# probs: narray, [#sample, #class], softmax(outputs of the last fc layer)
def dpn_expected_entropy(alphas, epsilon=1e-8):
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    alpha_0 = np.sum(alphas, axis=1, keepdims=True)
    expected_entropy = -np.sum(np.exp(np.log(alphas) - np.log(alpha_0)) * (digamma(alphas + 1.0) - digamma(alpha_0 + 1.0)), axis=1)
    return expected_entropy

def dpn_differential_entropy(alphas, epsilon=1e-8):
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    diff_entropy = np.asarray([dirichlet(alpha).entropy() for alpha in alphas])
    return diff_entropy

def dpn_entropy_expected(probs, epsilon=1e-8):
    probs = np.asarray(probs, dtype=np.float64) + epsilon
    entropy_expected = np.asarray([entropy(prob) for prob in probs], dtype=np.float64)
    return entropy_expected

def dpn_max_probability(probs, epsilon=1e-8):
    max_pred_prob = np.max(probs, axis=1)
    return max_pred_prob

def dpn_mutual_information(probs, alphas, epsilon=1e-8):
    exp_ent = dpn_expected_entropy(alphas, epsilon=epsilon)
    ent_exp = dpn_entropy_expected(probs, epsilon=epsilon)
    return ent_exp - exp_ent

def dpn_uncertainty(probs, alphas, epsilon=1e-8):
    entropy_expected = dpn_entropy_expected(probs)
    max_prob = dpn_max_probability(probs)
    mi = dpn_mutual_information(probs, alphas, epsilon)
    diff_ent = dpn_differential_entropy(alphas)
    return max_prob, entropy_expected, mi, diff_ent

def dpn_auc(probs, alphas, targets):
    max_prob, ent, mi, diff_ent = dpn_uncertainty(probs, alphas)
    predictions = np.argmax(probs, axis=1)
    rightwrong = np.asarray(targets != predictions, dtype=np.int32)
    return (roc_auc_score(rightwrong, 1. - max_prob), roc_auc_score(rightwrong, ent), 
           roc_auc_score(rightwrong, mi), roc_auc_score(rightwrong, diff_ent), 
           aupr(rightwrong, 1. - max_prob), aupr(rightwrong, ent), 
           aupr(rightwrong, mi), aupr(rightwrong, diff_ent))

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

def gmm_expected_entropy(alphas, epsilon=1e-8):
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    alpha_0 = np.sum(alphas, axis=1, keepdims=True)
    expected_entropy = -np.sum(np.exp(np.log(alphas) - np.log(alpha_0)) * (digamma(alphas + 1.0) - digamma(alpha_0 + 1.0)), axis=1)
    return expected_entropy

def gmm_mutual_information(probs, alphas, epsilon=1e-8):
    exp_ent = gmm_expected_entropy(alphas, epsilon=epsilon)
    ent_exp = gmm_entropy_expected(probs, epsilon=epsilon)
    return ent_exp - exp_ent

def gmm_differential_entropy(alphas, epsilon=1e-8):
    diff_entropy = np.asarray([dirichlet(alpha + epsilon).entropy() for alpha in alphas])
    return diff_entropy

def gmm_uncertainty(probs, alphas, epsilon=1e-8):
    entropy_expected = gmm_entropy_expected(probs)
    max_prob = gmm_max_probability(probs)
    mi = gmm_mutual_information(probs, alphas, epsilon)
    diff_ent = gmm_differential_entropy(alphas)
    return max_prob, entropy_expected, mi, diff_ent

def gmm_auc(probs, alphas, targets):
    max_prob, ent, mi, diff_ent = gmm_uncertainty(probs, alphas)
    predictions = np.argmax(probs, axis=1)
    rightwrong = np.asarray(targets != predictions, dtype=np.int32)
    print(np.sum(rightwrong))
    # return (auroc(rightwrong, max_prob, pos_label=0), auroc(rightwrong, ent), 
    #        auroc(rightwrong, mi), auroc(rightwrong, diff_ent), 
    #        aupr(rightwrong, max_prob, pos_label=0), aupr(rightwrong, ent), 
    #        aupr(rightwrong, mi), aupr(rightwrong, diff_ent))
    return (roc_auc_score(rightwrong, 1. - max_prob), roc_auc_score(rightwrong, ent), 
           roc_auc_score(rightwrong, mi), roc_auc_score(rightwrong, diff_ent), 
           average_precision_score(rightwrong, 1. - max_prob), average_precision_score(rightwrong, ent), 
           average_precision_score(rightwrong, mi), average_precision_score(rightwrong, diff_ent))


