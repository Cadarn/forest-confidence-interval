import math
import numpy as np
from scipy.stats import mstats, norm
from scipy.optimize import minimize
from sklearn.ensemble.forest import _generate_sample_indices
import pandas as pd
from forestci.due import _due, _BibTeX

__all__ = ["calc_inbag", "random_forest_error", "_bias_correction",
           "_core_computation","_gfit","_gbayes","_calibrateEB"]

_due.cite(_BibTeX("""
@ARTICLE{Wager2014-wn,
  title       = "Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife",
  author      = "Wager, Stefan and Hastie, Trevor and Efron, Bradley",
  journal     = "J. Mach. Learn. Res.",
  volume      =  15,
  number      =  1,
  pages       = "1625--1651",
  month       =  jan,
  year        =  2014,}"""),
          description=("Confidence Intervals for Random Forests:",
                       "The Jackknife and the Infinitesimal Jackknife"),
          path='forestci')


def calc_inbag(n_samples, forest):
    """
    Derive samples used to create trees in scikit-learn RandomForest objects.

    Recovers the samples in each tree from the random state of that tree using
    :func:`forest._generate_sample_indices`.

    Parameters
    ----------
    n_samples : int
        The number of samples used to fit the scikit-learn RandomForest object.

    forest : RandomForest
        Regressor or Classifier object that is already fit by scikit-learn.

    Returns
    -------
    Array that records how many times a data point was placed in a tree.
    Columns are individual trees. Rows are the number of times a sample was
    used in a tree.
    """
    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    for t_idx in range(n_trees):
        #GradientBoostingRegressor outputs individual trees as 1-element numpy arrays, this gets around it
        random_state = forest.estimators_[t_idx].random_state if not isinstance(forest.estimators_[t_idx],np.ndarray) else forest.estimators_[t_idx][0].random_state
        sample_idx.append(
            _generate_sample_indices(random_state, n_samples))
        inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    return inbag

def _gbayes(x0, g_est, sigma):
    """
    Bayes posterior estimation with Gaussian noise.

    Parameters
    ----------
    x0 : num
        An observation.

    g_est: DataFrame
        A prior density (returned by _gfit).

    sigma: num
        Monte Carlo noise estimate.

    Returns
    -------
    Posterior estimate E[mu|x0].
    """
    Kx = norm.pdf((g_est.x - x0) / sigma)
    post = Kx * g_est.g
    post /= np.sum(post)

    return np.sum(post * g_est.x)

def _gfit(X, sigma, p = 2, nbin = 1000, unif_fraction = 0.1):
    """
    Fit empirical Bayes prior in the hierarchical model
    mu ~ G, X ~ N(mu,sigma^2)

    Parameters
    ----------
    X : ndarray
        A vector of observations.

    sigma : num
        Monte Carlo noise estimate.

    p : int
        Number of parameters used to fit G.

    nbin : int
        Number of bins used for discrete approximation.

    unif_fraction : num
        Fraction of G modeled as "slab"

    Returns
    -------
    Posterior density estimate G (DataFrame)
    """
    xvals = np.linspace(np.min([np.min(X) - 2 * np.std(X), 0]), np.max([np.max(X) + 2 * np.std(X), np.std(X)]), nbin)
    binw = xvals[1] - xvals[0]

    zero_idx = np.max(np.where(xvals <= 0))
    noise_kernel = norm.pdf(xvals / sigma) * binw / sigma

    if zero_idx > 1:
        noise_rotate = np.roll(noise_kernel,len(noise_kernel)-zero_idx)
    else:
        noise_rotate = noise_kernel

    XX = np.array([xvals**i*(xvals>=0) for i in np.arange(1,p+1)]).T

    def neg_loglik(eta):
        g_eta_raw = np.exp(np.dot(XX,eta)) * (xvals >= 0)
        if ((np.sum(g_eta_raw) == np.inf) | (np.sum(g_eta_raw) <= 100 * np.finfo(float).eps)):
            return 1000 * (len(X) + np.sum(eta**2))

        g_eta_main = g_eta_raw / np.sum(g_eta_raw)
        g_eta = (1 - unif_fraction) * g_eta_main + unif_fraction * (xvals >= 0) / np.sum(xvals >= 0)
        f_eta = np.convolve(g_eta, noise_rotate, mode = 'same')
        return np.sum(np.interp(X, xvals, -np.log(np.maximum(f_eta, 0.0000001))))

    eta_hat = minimize(neg_loglik, np.repeat(-1, p)).x
    g_eta_raw = np.exp(np.dot(XX,eta_hat)) * (xvals >= 0)
    g_eta_main = g_eta_raw / np.sum(g_eta_raw)
    g_eta = (1 - unif_fraction) * g_eta_main + unif_fraction * (xvals >= 0) / np.sum(xvals >= 0)

    return pd.DataFrame({'x': xvals, 'g': g_eta})

def _calibrateEB(V_IJ_unbiased, sigma2):
    """
    Empirical Bayes calibration of noisy variance estimates.

    Parameters
    ----------
    V_IJ_unbiased : ndarray
        List of variance estimates.

    sigma2 : num
        Estimate of Monte Carlo noise in V_IJ_unbiased.

    Returns
    -------
    Calibrated variance estimates.
    """
    if sigma2 <= 0 or np.min(V_IJ_unbiased) == np.max(V_IJ_unbiased):
        return(np.maximum(V_IJ_unbiased, 0))

    sigma = np.sqrt(sigma2)
    eb_prior = _gfit(V_IJ_unbiased, sigma)

    if len(V_IJ_unbiased >= 200):
        # If there are many test points, use interpolation to speed up computations
        calib_x = mstats.mquantiles(V_IJ_unbiased, prob = np.arange(0., 1.01, 0.02))
        calib_y = np.array([_gbayes(xx,eb_prior,sigma) for xx in calib_x])
        calib_all = np.interp(V_IJ_unbiased, calib_x, calib_y)
    else:
        calib_all = np.array([_gbayes(xx, eb_prior, sigma) for xx in V_IJ_unbiased])

    return calib_all

def _core_computation(X_train, X_test, inbag, pred_centered, n_trees):
    cov_hat = np.zeros((X_train.shape[0], X_test.shape[0]))

    for t_idx in range(n_trees):
        inbag_r = (inbag[:, t_idx] - 1).reshape(-1, 1)
        pred_c_r = pred_centered.T[t_idx].reshape(1, -1)
        cov_hat += np.dot(inbag_r, pred_c_r) / n_trees
    V_IJ = np.sum(cov_hat ** 2, 0)
    return V_IJ


def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    n_train_samples = inbag.shape[0]
    n_var = np.mean(np.square(inbag[0:n_trees]).mean(axis=1).T.view() -
                    np.square(inbag[0:n_trees].mean(axis=1)).T.view())
    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var / n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def random_forest_error(forest, inbag, X_train, X_test, calibrate = True, used_trees = 'all'):
    """
    Calculates error bars from scikit-learn RandomForest estimators.

    RandomForest is a regressor or classifier object
    this variance can be used to plot error bars for RandomForest objects

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    inbag : ndarray
        The inbag matrix that fit the data.

    X : ndarray
        An array with shape (n_sample, n_features).

    calibrate : bool
        Calibrate for MC effects, takes care of the negative values

    used_trees : 'all' or ndarray
        Use the full forest for the prediction or a subsample, with tree indices specified by the array.

    Returns
    -------
    An array with the unbiased sampling variance (V_IJ_unbiased)
    for a RandomForest object.

    See Also
    ----------
    :func:`calc_inbag`

    Notes
    -----
    The calculation of error is based on the infinitesimal jackknife variance,
    as described in [Wager2014]_ and is a Python implementation of the R code
    provided at: https://github.com/swager/randomForestCI

    .. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
    """

    n_trees = forest.n_estimators

    if used_trees == 'all':
        pred = np.array([tree.predict(X_test) for tree in forest]).T
        pred_mean = np.mean(pred, 0)
        pred_centered = pred - pred_mean
        V_IJ = _core_computation(X_train, X_test, inbag, pred_centered, n_trees)
        V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)
    else:
        pred = np.array([forest[i].predict(X_test) for i in used_trees]).T
        pred_mean = np.mean(pred, 0)
        pred_centered = pred - pred_mean
        V_IJ = _core_computation(X_train, X_test, inbag[:,used_trees], pred_centered, len(used_trees))
        V_IJ_unbiased = _bias_correction(V_IJ, inbag[:,used_trees], pred_centered, len(used_trees))

    if calibrate:
        # Compute variance estimates using half the trees
        calibration_ratio = 2
        n_samples = math.ceil(n_trees / calibration_ratio)

        results_ss = random_forest_error(forest, inbag, X_train, X_test, calibrate = False,\
                                        used_trees = np.random.choice(n_trees,n_samples))

        # Use this second set of variance estimates to estimate scale of Monte Carlo noise
        sigma2_ss = np.mean((results_ss - V_IJ_unbiased)**2)
        delta = n_samples / n_trees
        sigma2 = (delta**2 + (1 - delta)**2) / (2 * (1 - delta)**2) * sigma2_ss

        # Use Monte Carlo noise scale estimate for empirical Bayes calibration
        vars_calibrated = _calibrateEB(V_IJ_unbiased, sigma2)
        V_IJ_unbiased = vars_calibrated

    return V_IJ_unbiased
