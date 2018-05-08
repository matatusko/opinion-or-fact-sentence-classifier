import numpy as np
from scipy.stats import distributions
from scipy.stats import binom


def mcnemar_table(y_target, y_model1, y_model2):
    """
    Compute a 2x2 contigency table for McNemar's test.
    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.
    y_model1 : array-like, shape=[n_samples]
        Predicted class labels from model as 1D NumPy array.
    y_model2 : array-like, shape=[n_samples]
        Predicted class labels from model 2 as 1D NumPy array.
    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    """
    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')

    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number'
                         ' of elements.')

    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number'
                         ' of elements.')

    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)

    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true

    tb = np.zeros((2, 2), dtype=int)

    tb[0, 0] = np.sum(plus_true == 2)
    tb[1, 1] = np.sum(plus_true == 0)
    tb[1, 0] = np.sum(minus_true == 1)
    tb[0, 1] = np.sum(minus_true == -1)

    return tb


def mcnemar(ary, corrected=True, exact=False):
    """
    McNemar test for paired nominal data
    Parameters
    -----------
    ary : array-like, shape=[2, 2]
        2 x 2 contigency table (as returned by evaluate.mcnemar_table),
        where
        a: ary[0, 0]: # of samples that both models predicted correctly
        b: ary[0, 1]: # of samples that model 1 got right and model 2 got wrong
        c: ary[1, 0]: # of samples that model 2 got right and model 1 got wrong
        d: aryCell [1, 1]: # of samples that both models predicted incorrectly
    corrected : array-like, shape=[n_samples] (default: True)
        Uses Edward's continuity correction for chi-squared if `True`
    exact : bool, (default: False)
        If `True`, uses an exact binomial test comparing b to
        a binomial distribution with n = b + c and p = 0.5.
        It is highly recommended to use `exact=True` for sample sizes < 25
        since chi-squared is not well-approximated
        by the chi-squared distribution!
    Returns
    -----------
    chi2, p : float or None, float
        Returns the chi-squared value and the p-value;
        if `exact=True` (default: `False`), `chi2` is `None`
    """

    if not ary.shape == (2, 2):
        raise ValueError('Input array must be a 2x2 array.')

    b = ary[0, 1]
    c = ary[1, 0]
    n = b + c

    if not exact:
        if corrected:
            chi2 = (abs(ary[0, 1] - ary[1, 0]) - 1.0)**2 / float(n)
        else:
            chi2 = (ary[0, 1] - ary[1, 0])**2 / float(n)
        p = distributions.chi2.sf(chi2, 1)

    else:
        p = 2. * sum([binom.pmf(k=i, n=n, p=0.5, loc=0) for i in range(b, n)])
        chi2 = None

    return chi2, p

# LOAD REAL VALUES
real_values = load_pickle('./real_test_values.pickle')

# LOAD BOW PREDICTIONS
nb_preds_bow = load_pickle('bow_pickles/nb_preds_bow.pickle')
rf_preds_bow = load_pickle('bow_pickles/rf_preds_bow.pickle')
lr_preds_bow = load_pickle('bow_pickles/lr_preds_bow.pickle')
nn_preds_bow = load_pickle('bow_pickles/nn_preds_bow.pickle')
svm_preds_bow = load_pickle('bow_pickles/svm_preds_bow.pickle')

# LOAD EPOS PREDICTIONS
nb_preds_epos = load_pickle('epos_pickles/nb_preds_epos.pickle')
rf_preds_epos = load_pickle('epos_pickles/rf_preds_epos.pickle')
lr_preds_epos = load_pickle('epos_pickles/lr_preds_epos.pickle')
nn_preds_epos = load_pickle('epos_pickles/nn_preds_epos.pickle')
svm_preds_epos = load_pickle('epos_pickles/svm_preds_epos.pickle')

assert len(real_values) == len(nb_preds_bow) == len(rf_preds_bow) == len(lr_preds_bow) == len(nn_preds_bow) == len(svm_preds_bow) == len(nb_preds_epos) == len(rf_preds_epos) == len(lr_preds_epos) == len(nn_preds_epos) == len(svm_preds_epos)

mc_table_nb = mcnemar_table(real_values, nb_preds_epos, nb_preds_bow)
mc_nb = mcnemar(mc_table_nb)
# (214.87213114754098, 1.1883965893410858e-48)

mc_table_rf = mcnemar_table(real_values, rf_preds_epos, rf_preds_bow)
mc_rf = mcnemar(mc_table_rf)
# (54.339622641509436, 1.6866636496614973e-13)

mc_table_lr = mcnemar_table(real_values, lr_preds_epos, lr_preds_bow)
mc_lr = mcnemar(mc_table_lr)
# (189.22499999999999, 4.6935189317142255e-43)

mc_table_nn = mcnemar_table(real_values, nn_preds_epos, nn_preds_bow)
mc_nn = mcnemar(mc_table_nn)
# (66.125, 4.2321366174257358e-16)

mc_table_svm = mcnemar_table(real_values, svm_preds_epos, svm_preds_bow)
mc_svm = mcnemar(mc_table_svm)
# (809.2663507109005, 5.2167985194144989e-178)