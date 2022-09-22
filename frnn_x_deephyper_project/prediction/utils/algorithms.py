import os
import json
import numpy as np
import tensorflow as tf
import time
import gpflow
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from .managing import average_prediction
from .evaluation import get_stats, compute_auc_from_stats


class ReaderModel(BaseEstimator, ClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X


def create_calibrator(y_true, y_pred, sample_weight=None):
    y_true_flat = np.concatenate(y_true).flatten()
    y_pred_flat = np.concatenate(y_pred).flatten()
    y_pred_stack = np.stack([y_pred_flat, y_pred_flat], axis=1)

    reader = ReaderModel()
    # wrap the model
    calibrator = CalibratedClassifierCV(reader, method='sigmoid')
    calibrator.fit(y_pred_stack, y_true_flat, sample_weight=sample_weight)
    return calibrator


def get_calibrator(y_pred, y_true, dataset, subset, criteria, model, group=None, calibrator=None, sample_weight=None):
    group = "" if group is None else f"{group}/"
    calibrator = "" if calibrator is None else f"{calibrator}/"
    path = f"calibration/calibrators/{dataset}/{criteria}/{group}{calibrator}{model}.p"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            calibrator = pickle.load(f)
    elif criteria == subset:
        calibrator = create_calibrator(y_true, y_pred, sample_weight=sample_weight)
        with open(path, 'wb') as f:
            pickle.dump(calibrator, f)
    else:
        raise ValueError(f"Calibrator does not exist and criteria {criteria} does not match subset {subset}.")
    return calibrator


def apply_calibration_model(y_pred, y_true, dataset, subset, criteria, model, group=None, calibrator="sigmoid", sample_weight=None):
    calibrator = get_calibrator(y_pred, y_true, dataset, subset, criteria, model, group=group, calibrator=calibrator, sample_weight=sample_weight)
    for i, y in enumerate(y_pred):
        y_flat = y.flatten()
        y_pred[i] = np.expand_dims((calibrator.predict_proba(np.stack([y_flat, y_flat], axis=1)))[:, 1], axis=-1)
    return y_pred


def apply_balanced_calibration_model(y_pred, y_true, dataset, subset, criteria, model, group=None):
    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    N = len(disruptive)
    y_true_clip = np.clip(np.concatenate(y_true).flatten(), 0, 1)
    n_p = np.sum(disruptive)
    n_n = N - n_p
    coeff_p = N/(2*n_p)
    coeff_n = N/(2*n_n)
    sample_weight = y_true_clip*coeff_p + (1-y_true_clip)*coeff_n
    y_pred = apply_calibration_model(y_pred, y_true, dataset, subset, criteria, model, group=group, calibrator="balanced_sigmoid", sample_weight=sample_weight)
    return y_pred


def apply_calibration_models(y_pred_models, y_true, dataset, subset, criteria, group, models):
    for idx, model in enumerate(models):
        y_pred = [y[idx, :] for y in y_pred_models]
        y_pred = apply_calibration_model(y_pred, y_true, dataset, subset, criteria, model, group=group, calibrator="sigmoid")
        for i, y in enumerate(y_pred_models):
            y[idx, :] = y_pred[i]
    return y_pred_models


def apply_balanced_calibration_models(y_pred_models, y_true, dataset, subset, criteria, group, models):
    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    N = len(disruptive)
    y_true_clip = np.clip(np.concatenate(y_true).flatten(), 0, 1)
    n_p = np.sum(disruptive)
    n_n = N - n_p
    coeff_p = N/(2*n_p)
    coeff_n = N/(2*n_n)
    sample_weight = y_true_clip*coeff_p + (1-y_true_clip)*coeff_n

    for idx, model in enumerate(models):
        y_pred = [y[idx, :] for y in y_pred_models]
        y_pred = apply_calibration_model(y_pred, y_true, dataset, subset, criteria, model, group=group, calibrator="balanced_sigmoid", sample_weight=sample_weight)
        for i, y in enumerate(y_pred_models):
            y[idx, :] = y_pred[i]
    return y_pred_models


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def loss_balanced_crossentropy(y_true, y_pred):
    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    N = len(disruptive)
    y_true_clip = np.clip(np.concatenate(y_true).flatten(), 0, 1)
    y_pred_flat = np.concatenate(y_pred).flatten()
    n_p = np.sum(disruptive)
    n_n = N - n_p
    coeff_p = N/(2*n_p)
    coeff_n = N/(2*n_n)
    sample_weight = y_true_clip*coeff_p + (1-y_true_clip)*coeff_n
    loss = tf.keras.backend.binary_crossentropy(y_true_clip, y_pred_flat)
    balanced_loss = tf.math.reduce_mean(tf.math.multiply(loss, sample_weight))
    return balanced_loss


def loss_auc(y_true, y_pred):
    stats = get_stats(y_true, y_pred)
    auc = compute_auc_from_stats(stats)
    return -auc


def greedy_caruana(loss, y_true, y_pred_models, k):
    n_models = y_pred_models[0].shape[0]

    losses = [
        loss(
            y_true,
            average_prediction(
                y_pred_models, idx=[i]
            ),
        )
        for i in range(n_models)  # iterate over all models
    ]
    i_min = np.nanargmin(losses)
    loss_min = losses[i_min]
    ensemble_members = [i_min]
    print(f"Eval: {loss_min:.3f} - Ensemble: {ensemble_members}")

    while len(np.unique(ensemble_members)) < k:
        losses = [
            loss(
                y_true,
                average_prediction(
                    y_pred_models, idx=ensemble_members + [i]
                ),
            )
            for i in range(n_models)  # iterate over all models
        ]
        i_min_ = np.nanargmin(losses)
        loss_min_ = losses[i_min_]

        if loss_min_ < loss_min:
            if (
                len(np.unique(ensemble_members)) == 1 and ensemble_members[0] == i_min_
            ):  # numerical errors...
                return [int(member) for member in ensemble_members]
            loss_min = loss_min_
            ensemble_members.append(i_min_)
            print(f"Eval: {loss_min:.3f} - Ensemble: {ensemble_members}")
        else:
            print(ensemble_members)
            print([int(member) for member in ensemble_members])
            return [int(member) for member in ensemble_members]

    return ensemble_members


def apply_caruana(loss, y_pred_models, y_true, k, dataset, subset, criteria, calibrator, group):
    path = f"calibration/caruana_ensembles/{dataset}/{criteria}/{group}/ensembles_{calibrator}.json"
    if os.path.exists(path):
        with open(path,"r") as f:
            ensembles = json.load(f)
    else:
        ensembles = dict()
    if f"{k}" in ensembles.keys():
        ensemble = ensembles[f"{k}"]
    elif subset == criteria:
        ensemble = greedy_caruana(loss, y_true, y_pred_models, k)
        ensembles[f"{k}"] = ensemble
        with open(path, "w") as f:
            json.dump(ensembles, f)
    else:
        raise ValueError(f"Ensemble does not exist and criteria {criteria} does not match subset {subset}.")
    return ensemble


def gradient(y_pred_models, y_true, keep_model_thresh=0):
    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    y_pred_models = np.concatenate(y_pred_models, axis=1)
    shape = y_pred_models.shape
    M = shape[0]
    N = len(disruptive)
    y_pred_models = y_pred_models.reshape((M, shape[1]))
    y_pred_models = tf.convert_to_tensor(y_pred_models)
    y_true = np.clip(np.concatenate(y_true).flatten(), 0, 1)
    n_p = np.sum(disruptive)
    n_n = N - n_p
    coeff_p = N/(2*n_p)
    coeff_n = N/(2*n_n)
    sample_weight = y_true*coeff_p + (1-y_true)*coeff_n

    def ensemble_calib(y_pred, alphas, betas, gammas):
        y_pred_calib = tf.einsum('i,ij->j', tf.math.softmax(gammas), tf.sigmoid(tf.math.add(tf.einsum('i,ij->ij', alphas, y_pred), tf.einsum('i,j->ij', betas, np.ones(y_true.shape[0])))))
        return y_pred_calib

    def optimize(y_true, y_pred_models, loss):
        alphas = np.ones(M, dtype='float64')
        betas = np.zeros(M, dtype='float64')
        gammas = np.ones(M, dtype='float64')/M
        ai = tf.Variable(alphas)
        bi = tf.Variable(betas)
        gi = tf.Variable(gammas)
        
        def loss_closure():
            y_pred = ensemble_calib(y_pred_models, ai, bi, gi)
            return tf.math.reduce_mean(tf.math.multiply(loss(y_true, y_pred), sample_weight))
        
        maxiter = 1000
        opt = gpflow.optimizers.Scipy()
        t = time.time()
        result = opt.minimize(
            loss_closure,
            [ai, bi, gi],
            method="L-BFGS-B",
            tol=1e-8,
            options=dict(disp=False, maxiter=maxiter),
            compile=True,
        )
        print(f"Calibrator's optimization took {time.time()-t:.2f}s.")
        w_f = result["x"]
        
        return w_f

    x_f = optimize(y_true, y_pred_models, tf.keras.backend.binary_crossentropy)
    raw_alphas = [x_f[i] for i in range(M)]
    raw_betas = [x_f[i] for i in range(M, 2*M)]
    raw_gammas = [x_f[i] for i in range(2*M, 3*M)]
    raw_gammas_prime = list(np.exp(raw_gammas)/np.sum(np.exp(raw_gammas)))
    models_to_keep = [gamma_prime > keep_model_thresh for gamma_prime in raw_gammas_prime]

    alphas = list(alpha for alpha, keep in zip(raw_alphas, models_to_keep) if keep)    
    betas = list(beta for beta, keep in zip(raw_betas, models_to_keep) if keep)
    gammas = list(gamma for gamma, keep in zip(raw_gammas, models_to_keep) if keep)
    gammas_prime = list(np.exp(gammas)/np.sum(np.exp(gammas)))
    ensemble = list(i for i, keep in enumerate(models_to_keep) if keep)

    print(f"alphas: {', '.join([f'{a:.2f}' for a in alphas])}")
    print(f"betas: {', '.join([f'{b:.2f}' for b in betas])}")
    print(f"gammas_prime: {', '.join([f'{g:.2f}' for g in gammas_prime])}")
    params = dict(
        alphas=alphas,
        betas=betas,
        gammas=gammas_prime,
    )
    return params, ensemble


def test_other_gradient(y_pred_models, y_true):
    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    y_pred_models = np.concatenate(y_pred_models, axis=1)
    shape = y_pred_models.shape
    M = shape[0]
    N = len(disruptive)
    y_pred_models = y_pred_models.reshape((M, shape[1]))
    y_pred_models = tf.convert_to_tensor(y_pred_models)
    y_true = np.clip(np.concatenate(y_true).flatten(), 0, 1)
    n_p = np.sum(disruptive)
    n_n = N - n_p
    coeff_p = N/(2*n_p)
    coeff_n = N/(2*n_n)
    sample_weight = y_true*coeff_p + (1-y_true)*coeff_n

    def ensemble_calib(y_pred, alphas, betas, weights):
        y_c_out = tf.sigmoid(tf.math.add(tf.einsum('i,ij->ij', alphas, y_pred), tf.einsum('i,j->ij', betas, np.ones(y_true.shape[0]))))
        y_c_conf = y_c_out
        conf = tf.math.maximum(y_c_conf, tf.math.subtract(np.ones(1, dtype='float64'), y_c_conf))
        gammas = tf.nn.softmax(tf.einsum('ik,ij->ij', tf.reshape(weights, (M, M)), conf), axis=0)
        y_models = tf.math.multiply(gammas, y_c_out)
        y_ens = tf.einsum('ij,ij->j', gammas, y_c_out)
        return y_ens

    def optimize(y_true, y_pred_models, loss):
        alphas = np.ones(M, dtype='float64')
        betas = np.zeros(M, dtype='float64')
        weights = np.eye(M, dtype='float64')/M
        ai = tf.Variable(alphas)
        bi = tf.Variable(betas)
        wi = tf.Variable(weights)
        
        def loss_closure():
            y_pred = ensemble_calib(y_pred_models, ai, bi, wi)
            return tf.math.reduce_mean(tf.math.multiply(loss(y_true, y_pred), sample_weight))
        
        maxiter = 1000
        opt = gpflow.optimizers.Scipy()
        t = time.time()
        result = opt.minimize(
            loss_closure,
            [ai, bi, wi],
            method="L-BFGS-B",
            tol=1e-8,
            options=dict(disp=False, maxiter=maxiter),
            compile=True,
        )
        print(f"Calibrator's optimization took {time.time()-t:.2f}s.")
        w_f = result["x"]
        
        return w_f

    x_f = optimize(y_true, y_pred_models, tf.keras.backend.binary_crossentropy)
    alphas = [x_f[i] for i in range(M)]
    betas = [x_f[i] for i in range(M, 2*M)]
    weights = [x_f[i] for i in range(2*M, 2*M+M**2)]
    weights = [[weights[i+M*j] for i in range(M)] for j in range(M)]

    print(f"alphas: {', '.join([f'{a:.2f}' for a in alphas])}")
    print(f"betas: {', '.join([f'{b:.2f}' for b in betas])}")
    print(f"weights: {weights}")
    params = dict(
        alphas=alphas,
        betas=betas,
        weights=weights,
    )
    return params


def apply_calibration_ensemble(y_pred_models, y_true, dataset, subset, criteria, group, keep_models_thresh=0):
    M = y_pred_models[0].shape[0]
    path = f"calibration/gradient/{dataset}/{criteria}/{group}.json"
    if os.path.exists(path):
        with open(path,"r") as f:
            all_params = json.load(f)
    else:
        all_params = dict()
    if f"{M}" in all_params.keys():
        params = all_params[f"{M}"]['params']
        ensemble = all_params[f"{M}"]['ensemble']
    elif subset == criteria:
        params, ensemble = gradient(y_pred_models, y_true, keep_model_thresh=keep_models_thresh)
        all_params[f"{M}"] = dict(
            params=params,
            ensemble=ensemble
        )
        with open(path, "w") as f:
            json.dump(all_params, f)
    else:
        raise ValueError(f"Parameters do not exist and criteria {criteria} does not match subset {subset}.")
    alphas = params['alphas']
    betas = params['betas']
    gammas = params['gammas']
    for i, idx in enumerate(ensemble):
        for y in y_pred_models:
            y[idx, :] = 1/(1+np.exp(-(alphas[i]*y[idx, :]+betas[i])))
    return y_pred_models, gammas, ensemble


def test_apply_calibration_ensemble(y_pred_models, y_true, dataset, subset, criteria, group):
    M = y_pred_models[0].shape[0]
    path = f"calibration/new_method/{dataset}/{criteria}/{group}.json"
    if os.path.exists(path):
        with open(path,"r") as f:
            all_params = json.load(f)
    else:
        all_params = dict()
    if subset == criteria:
        params = test_other_gradient(y_pred_models, y_true)
        all_params[f"{M}"] = params
        with open(path, "w") as f:
            json.dump(all_params, f)
    elif f"{M}" in all_params.keys():
        params = all_params[f"{M}"]
    else:
        raise ValueError(f"Parameters do not exist and criteria {criteria} does not match subset {subset}.")

    alphas=np.array(params['alphas'])
    betas=np.array(params['betas'])
    weights=np.array(params['weights'])

    sample_weights = []    
    for i, y in enumerate(y_pred_models):
        y_c_out = 1/(1+np.exp(-(np.einsum('i,ij...->ij...', alphas, y) + np.expand_dims(np.einsum('i,j->ij', betas, np.ones(y.shape[1])), axis=-1))))
        y_c_conf = y_c_out
        conf = np.maximum(y_c_conf, 1-y_c_conf)
        weights_conf = np.einsum('ik,ij...->ij...', weights, conf)
        gammas = np.exp(weights_conf)/np.sum(np.exp(weights_conf), axis=0)
        y_pred_models[i] = y_c_out
        sample_weights.append(gammas)
    return y_pred_models, sample_weights

