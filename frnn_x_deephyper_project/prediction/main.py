import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import yaml
import pickle
import gpflow
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from plasma.utils.performance import PerformanceAnalyzer

print("GOING")

width = 5
height = width / 1.618

matplotlib.rcParams.update({
    'font.size': 12,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 72,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})


class ReaderModel(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X


class ReaderMultiModel(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        return X.mean(axis=0)

    def predict_proba(self, X):
        return X.mean(axis=0)


def aggregate_predictions(y_pred, idx=None, weights=None):
    agg_y_pred = []
    if idx is None:
        for pred in y_pred:
            agg_y_pred.append(np.average(pred, weights=weights, axis=0))
    else:
        for pred in y_pred:
            agg_y_pred.append(np.average(pred[idx, :], weights=weights, axis=0))
    return agg_y_pred


def greedy_caruana(loss, y_true, y_pred, k=2, verbose=0):
    n_models = np.shape(y_pred[0])[0]

    losses = [
        loss(
            y_true,
            aggregate_predictions(
                y_pred, idx=[i]
            ),
        )
        for i in range(n_models)  # iterate over all models
    ]
    i_min = np.nanargmin(losses)
    loss_min = losses[i_min]
    ensemble_members = [i_min]
    if verbose:
        print(f"Eval: {loss_min:.3f} - Ensemble: {ensemble_members}")

    while len(np.unique(ensemble_members)) < k:
        losses = [
            loss(
                y_true,
                aggregate_predictions(
                    y_pred, idx=ensemble_members + [i]
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
                return ensemble_members
            loss_min = loss_min_
            ensemble_members.append(i_min_)
            if verbose:
                print(f"Eval: {loss_min:.3f} - Ensemble: {ensemble_members}")
        else:
            return ensemble_members

    return ensemble_members


def load_data(dataset, subset, key, group=None):
    if group is None:
        group = ""
    else:
        group = f"{group}/"
    data = np.load(f"{dataset}/{subset}/{group}{key}.npz")
    return list(data.values())

def load_pred_models(dataset, subset, criteria=None, group="top_models", k=None, ensemble=None):
    """Loads predictions of top_k models and returns y_pred of the form list(k, shot_length, 1)
    """
    if criteria is not None:
        with open(f"{dataset}/{criteria}/{group}/specs.yaml", 'r') as f:
            specs = yaml.safe_load(f)
        ordered_list = list(reversed(sorted([(specs[key]['auc'], key) for key in specs.keys()])))
    else:
        with open(f"{dataset}/valid/{group}/specs.yaml", 'r') as f:
            specs = yaml.safe_load(f)
        ordered_list = list(sorted([(int(key.split('_')[1]), key) for key in specs.keys()]))
    y_true = load_data(dataset, subset, "y_gold")
    pred_list = [[] for _ in range(len(y_true))]
    models = []
    if ensemble is not None:
        idx_range = ensemble
    elif k is not None:
        idx_range = range(k)
    else:
        idx_range = range(len(specs))
    for i in idx_range:
        order, model = ordered_list[i]
        models.append(model)
        model_preds = load_data(dataset, subset, model, group=group)
        for model_pred, pred in zip(model_preds, pred_list):
            pred.append(model_pred)
    for i, pred in enumerate(pred_list):
        pred_list[i] = np.stack(pred, axis=0)
    
    return pred_list, models


def apply_calibration_model(y_pred, cal_type, dataset, criteria, model, group=None):
    if group is None:
        group = ""
    else:
        group = f"{group}/"
    with open(f"{dataset}/{criteria}/calibrators/{group}{cal_type}/{model}.p", 'rb') as f:
        calibrator = pickle.load(f)
    for i, y in enumerate(y_pred):
        y_flat = y.flatten()
        y_pred[i] = np.expand_dims((calibrator.predict_proba(np.stack([y_flat, y_flat], axis=1)))[:, 1], axis=-1)

print("STILL GOING")

def apply_calibration_ensemble(y_pred_ensemble, y_true):
    # y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    # y_pred_models = np.array(
    #     [
    #         [-2, -2, -2, -3, 3, -2, 3],
    #         [-3, -3, -3, -8, 8, 8, -3],
    #         [-1, -1, 2, -1, 2, -1, 2],
    #     ], dtype='float32'
    # )

    disruptive = np.array([1 in t for t in y_true], dtype='int32')
    y_pred_models = np.concatenate(y_pred_ensemble, axis=1)
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

    print(f"y_pred_models.shape: {y_pred_models.shape}")
    print(f"y_true.shape: {y_true.shape}")
    print(f"sample_weight.shape: {sample_weight.shape}")

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
    alphas = [x_f[i] for i in range(M)]
    betas = [x_f[i] for i in range(M, 2*M)]
    gammas = [x_f[i] for i in range(2*M, 3*M)]
    gammas = np.exp(gammas)/np.sum(np.exp(gammas))
    print(f"alphas: {', '.join([f'{a:.2f}' for a in alphas])}")
    print(f"betas: {', '.join([f'{b:.2f}' for b in betas])}")
    print(f"gammas: {', '.join([f'{g:.2f}' for g in gammas])}")

    for i in range(M):
        for y in y_pred_ensemble:
            y[i, :] = 1/(1+np.exp(-(alphas[i]*y[i, :]+betas[i])))
    return gammas

print("GOING ON AND ON")

def old_apply_calibration_ensemble(y_pred_ensemble, cal_type, dataset, criteria, group, models):
    with open(f"{dataset}/{criteria}/calibrators/ensembles/{cal_type}.p", 'rb') as f:
        calibrator_ens = pickle.load(f)
    for idx, model in enumerate(models):
        with open(f"{dataset}/{criteria}/calibrators/{group}/{cal_type}/{model}.p", 'rb') as f:
            calibrator = pickle.load(f)
        for y in y_pred_ensemble:
            y_flat = y[idx, :].flatten()
            y[idx, :] = np.expand_dims((calibrator.predict_proba(np.stack([y_flat, y_flat], axis=1)))[:, 1], axis=-1)
            # y_flat = (calibrator.predict_proba(np.stack([y_flat, y_flat], axis=1)))[:, 1]
            # y[idx, :] = np.expand_dims((calibrator_ens.predict_proba(np.stack([y_flat, y_flat], axis=1)))[:, 1], axis=-1)


analyser = PerformanceAnalyzer(T_min_warn=30 , T_max_warn=1024, ignore_timesteps=100)

def get_stats(y_true, y_pred):
    thresh_range, accuracy_range, precision_range, tp_rate_range, fp_rate_range, tp_range, fp_range = analyser.get_metrics_vs_p_thresh_dh(y_pred, y_true, early_pred_counts=True)
    stats = dict(
        thresh_range=thresh_range,
        accuracy_range=accuracy_range,
        precision_range=precision_range,
        tp_rate_range=tp_rate_range,
        fp_rate_range=fp_rate_range,
        tp_range=tp_range,
        fp_range=fp_range,
    )
    return stats

def compute_auc_from_stats(stats):
    tp_rate_range = stats['tp_rate_range']
    fp_rate_range = stats['fp_rate_range']
    auc = analyser.roc_from_tp_fp_rates(tp_rate_range, fp_rate_range)
    return auc

def loss_auc(y_true, y_pred):
    stats = get_stats(y_true, y_pred)
    auc = compute_auc_from_stats(stats)
    return -auc

def loss_crossentropy(y_true, y_pred):
    None

def loss_balanced_crossentropy(y_true, y_pred):
    None

def compute_auc_original(y_true, y_pred):
    auc = analyser.get_roc_area(y_pred, y_true, None)
    return auc

def quick_plot_auc(stats, plot_kwargs):
    tp_rate_range = stats['tp_rate_range']
    fp_rate_range = stats['fp_rate_range']
    plt.plot(fp_rate_range, tp_rate_range, **plot_kwargs)

def plot_multi_auc(plot_data, subset, calibrator):
    plt.figure()
    for data in plot_data.values():
        stats = data['stats']
        plot_kwargs = data['plot_kwargs']
        quick_plot_auc(stats, plot_kwargs)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.legend(list(plot_data.keys()))
    plt.ylabel("True Positives Ratio")
    plt.xlabel("False Positives Ratio")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/roc/{calibrator}/roc_curve_{subset}.png")

def compute_uq_old(y_pred_models, weights=None):
    alea = []
    epis = []
    tota = []
    for p in y_pred_models:
        var_alea = np.average(p*(1-p), axis=0, weights=weights)
        m = np.average(p, axis=0, weights=weights)
        var_epis = np.average((p-m)**2, axis=0, weights=weights)
        var_tota = m*(1-m)
        alea.append(var_alea/0.25)
        epis.append(var_epis/0.25)
        tota.append(var_tota/0.25)
    return alea, epis, tota


def compute_uq(y_pred_models, weights=None):
    alea = []
    epis = []
    tota = []
    for p in y_pred_models:
        m = np.average(p, axis=0, weights=weights)
        min_p = np.minimum(p, 1-p)
        var_alea = np.average(min_p, axis=0, weights=weights)
        var_epis = np.average(p*(m<0.5) + (1-p)*(m>=0.5) - min_p, axis=0, weights=weights)
        var_tota = np.minimum(m, 1-m)
        alea.append(var_alea/0.5)
        epis.append(var_epis/0.5)
        tota.append(var_tota/0.5)
    return alea, epis, tota


def compute_uq_comp(y_pred_models, weights=None):
    var = []
    lin = []
    for p in y_pred_models:
        m = np.average(p, axis=0, weights=weights)
        var.append(m*(1-m)/0.25)
        lin.append(np.minimum(m, 1-m)/0.5)
    return var, lin


def plot_pred_shot(ax, y_pred, y_true, title=None):
    true_color = 'navy'
    # y_pred = y_pred/(np.max(np.abs(y_pred)))
    # alarm_time = get_alarm_time(y_pred)
    shot_length = len(y_true)

    ax.set_title(title)
    ax.plot(y_true, color=true_color)
    ax.plot(y_pred, color='black')
    ax.fill_between(list(range(shot_length)), y_true.flatten(), 0.5, color=true_color, alpha=0.5)
    # if alarm_time is not None:
    #     ax.plot(alarm_time, 0.02, marker='^', color='black', markersize=10, zorder=100)
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length])
    ax.legend(["y_true", "y_pred"], loc='upper left')


def plot_uq_shot(ax, var_ale, var_epi, var_tot):
    shot_length = len(var_tot)
    ax.plot(var_tot, color='black')
    ax.fill_between(list(range(shot_length)), var_epi.flatten(), color='fuchsia')
    ax.fill_between(list(range(shot_length)), (var_epi+var_ale).flatten(), var_epi.flatten(), color='cyan')
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, shot_length])
    ax.legend(["var_tot", "var_epi", "var_ale"], loc='upper left')

def plot_uq_comp_shot(ax, var_uq, lin_uq):
    shot_length = len(var_uq)
    ax.plot(var_uq, color='navy')
    ax.plot(lin_uq, color='crimson')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length])
    ax.legend(["p*(1-p)", "min(p, 1-p)"], loc='upper left')

def plot_tot_uq_shot(ax, var_tot):
    shot_length = len(var_tot)
    ax.plot(var_tot, color='black')
    ax.fill_between(list(range(shot_length)), var_tot.flatten(), color='crimson', alpha=0.5)
    ax.set_ylim([0, 0.26])
    ax.set_xlim([0, shot_length])
    ax.legend(["var_tot"], loc='upper left')

def plot_pred_uq_comp_shot(y_pred_list, y_true_list, var_uq_list, lin_uq_list, subset, calibrator, idx):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    plot_uq_comp_shot(ax[0], var_uq_list[idx], lin_uq_list[idx])
    ax[0].set_title(f"{subset} - shot {idx}")
    plot_pred_shot(ax[1], y_pred_list[idx], y_true_list[idx])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/uq_comp/{subset}/shot_{idx}.png")
    plt.clf()

def plot_pred_uq_shot(y_pred_list, y_true_list, var_ale_list, var_epi_list, var_tot_list, subset, calibrator, idx):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    plot_uq_shot(ax[0], var_ale_list[idx], var_epi_list[idx], var_tot_list[idx])
    ax[0].set_title(f"{subset} - shot {idx}")
    plot_pred_shot(ax[1], y_pred_list[idx], y_true_list[idx])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/uq/{subset}/{calibrator}/shot_{idx}.png")
    plt.clf()

def plot_pred_uq_model_shot(y_pred_list, y_true_list, var_tot_list, subset, calibrator, idx):
    plt.figure()
    fig, ax = plt.subplots(2)
    plot_tot_uq_shot(ax[0], var_tot_list[idx])
    ax[0].set_title(f"{subset} - shot {idx}")
    plot_pred_shot(ax[1], y_pred_list[idx], y_true_list[idx])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/pred/{subset}/{calibrator}/shot_{idx}.png")
    plt.clf()

def plot_compare_shot(y_pred_list_list, y_true_list, model_labels, subset, calibrator, idx):
    plt.figure()
    fig, ax = plt.subplots(len(model_labels), sharex=True)
    for i, y_pred_list in enumerate(y_pred_list_list):
        plot_pred_shot(ax[i], y_pred_list[idx], y_true_list[idx], title=model_labels[i])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/compare/{subset}/{calibrator}/shot_{idx}.png")
    plt.clf()


if __name__ == "__not_main__":
    dataset = "predictions/d3d_2019"
    subset = "test"
    criteria = "valid"
    group = "top_models"
    model = "top_8"
    calibrator = "sigmoid"
    
    print("\nDATA_LOADING")
    y_pred = load_data(dataset, subset, model, group=group)
    y_true = load_data(dataset, subset, "y_gold")

    print("\nCALIBRATION")
    if calibrator in ["sigmoid", "isotonic"]:
        print(f"applying {calibrator} calibration..")
        apply_calibration_model(y_pred, calibrator, dataset, criteria, model, group=group)
    else:
        print(f"applying basic sigmoid..")
        calibrator = "basic"
        for i, pred in enumerate(y_pred):
            y_pred[i] = 1/(1+np.exp(-pred))

    var_tota = []
    for p in y_pred:
        var_tota.append(p*(1-p))
    
    for idx in range(min(1000, len(y_true))):
        plot_pred_uq_model_shot(y_pred, y_true, var_tota, subset, calibrator, idx)


if __name__ == "__main__":
    #####################################           MAIN          #####################################
    print("\nMAIN")

    dataset = "predictions/d3d_2019"
    subset = "valid"
    group = "top_models" # "top_models", "balanced_models"
    criteria = "valid"
    calibrator = "sigmoid" # "sigmoid", "isotonic", "basic"
    print(f"dataset: {dataset}/{subset} - criteria: {criteria} - calibrator: {calibrator}")
    top = 20
    k = 20
    top_ensembles = dict(
        sigmoid=[0, 1, 8, 2, 37, 13, 4, 12, 13, 37],
        isotonic=[0, 1, 2, 13, 12, 37, 8, 4, 13, 1, 36],
        # basic=[0, 1, 8, 2, 4, 13, 4],
    )
    ensembles = dict(
        top_models=top_ensembles,
    )
    ensemble = list(range(top)) #ensembles.get(group, top_ensembles).get(calibrator, None)

    #####################################       DATA_LOADING      #####################################
    print("\nDATA_LOADING")

    print("loading true values..")
    y_true = load_data(dataset, subset, "y_gold")
    print("loading baseline's pred..")
    y_pred_baseline = load_data(dataset, subset, "baseline")
    print("loading top model's pred..")
    y_pred_models, models = load_pred_models(dataset, subset, group=group, criteria=criteria, k=top, ensemble=ensemble)

    #####################################      PREPROCESSING      #####################################
    print("\nPREPROCESSING")

    # # MAX_ALARM
    # print("applying 'max_alarm' function..")
    # for i, pred in enumerate(y_pred_models):
    #     y_pred_models[i] = np.maximum.accumulate(pred)
    # for i, pred in enumerate(y_pred_baseline):
    #     y_pred_baseline[i] = np.maximum.accumulate(pred)
    
    # CALIBRATE
    if calibrator in ["sigmoid", "isotonic"]:
        print(f"applying {calibrator} calibration..")
        apply_calibration_model(y_pred_baseline, calibrator, dataset, criteria, "baseline")
        # old_apply_calibration_ensemble(y_pred_models, calibrator, dataset, criteria, group, models)
        # gammas = np.ones(len(models))/len(models)
        gammas = apply_calibration_ensemble(y_pred_models, y_true)
    else:
        print(f"applying basic sigmoid..")
        calibrator = "basic"
        for i, pred in enumerate(y_pred_models):
            y_pred_models[i] = 1/(1+np.exp(-pred))
        for i, pred in enumerate(y_pred_baseline):
            y_pred_baseline[i] = 1/(1+np.exp(-pred))
        gammas = np.ones(len(models))/len(models)

    #####################################    (TEST_EVAL_FUNC)     #####################################
    # print("\n(TEST_EVAL_FUNC)")

    # y_pred_best_model = aggregate_predictions(y_pred_models, idx=[0])
    # auc = compute_auc(y_true, y_pred_best_model)
    # auc_original = compute_auc_original(y_true, y_pred_best_model)
    # print(f"auc: {auc}")
    # print(f"auc_original: {auc_original}")
    # print("\napplying 'max_alarm' function..")
    # for i, pred in enumerate(y_pred_best_model):
    #     y_pred_best_model[i] = np.maximum.accumulate(pred)
    # auc = compute_auc(y_true, y_pred_best_model)
    # auc_original = compute_auc_original(y_true, y_pred_best_model)
    # print(f"auc: {auc}")
    # print(f"auc_original: {auc_original}")
    # assert 0 == 1

    #####################################      BEST_ENSEMBLE      #####################################
    print("\nBEST_ENSEMBLE")

    if ensemble is None:
        print("best_ensemble does not exist ; running Caruana algorithm..")
        best_ensemble = greedy_caruana(loss_crossentropy, y_true, y_pred_models, k=k, verbose=1)
        print(f"best_ensemble labels: {list(models[i] for i in best_ensemble)}")
    else:
        print("best_ensemble already exists :")
        best_ensemble = ensemble
        print(f"best_ensemble labels: {models}")
    print(f"best_ensemble idx: {best_ensemble}")

    #####################################       AGGREGATION       #####################################
    print("\nAGGREGATION")

    print("obtaining best_model's predictions..")
    y_pred_best_model = aggregate_predictions(y_pred_models, idx=[best_ensemble[0]])
    print("obtaining best_ensemble's predictions..")
    if ensemble is not None:
        best_ensemble = None
    y_pred_best_ensemble = aggregate_predictions(y_pred_models, idx=best_ensemble, weights=gammas)

    # flat_pred = (np.concatenate(y_pred_best_ensemble)).flatten()
    # flat_true = (np.concatenate(y_true)).flatten()
    # flat_pred = np.stack([flat_pred, flat_pred], axis=1)

    # from sklearn.base import BaseEstimator, ClassifierMixin
    # from sklearn.calibration import CalibratedClassifierCV


    # class ReaderModel(BaseEstimator, ClassifierMixin):

    #     def fit(self, X, y):
    #         self.classes_, y = np.unique(y, return_inverse=True)
    #         return self

    #     def predict(self, X):
    #         return X

    #     def predict_proba(self, X):
    #         return X

    # reader = ReaderModel()
    # # wrap the model
    # calibrated = CalibratedClassifierCV(reader, method='sigmoid')
    # calibrated.fit(flat_pred, flat_true)
    # path = f"{dataset}/{criteria}/calibrators/ensembles/sigmoid.p"
    # print("CALIBRATOR")
    # print(path)
    # with open(path, 'wb') as f:
    #     pickle.dump(calibrated, f)

    #####################################          OUTPUT         #####################################
    print("\nOUTPUT")

    print("computing stats..")
    baseline_stats = get_stats(y_true, y_pred_baseline)
    best_model_stats = get_stats(y_true, y_pred_best_model)
    best_ensemble_stats = get_stats(y_true, y_pred_best_ensemble)

    print("computing auc..")
    baseline_auc = compute_auc_from_stats(baseline_stats)
    best_model_auc = compute_auc_from_stats(best_model_stats)
    best_ensemble_auc = compute_auc_from_stats(best_ensemble_stats)
    print(f"baseline_auc: {baseline_auc}")
    print(f"best_model_auc: {best_model_auc}")
    print(f"best_ensemble_auc: {best_ensemble_auc}")

    # print("saving roc plot..")
    # plot_data = dict(
    #     baseline=dict(
    #         stats=baseline_stats,
    #         plot_kwargs=dict(
    #             color='navy',
    #             linestyle='-',
    #         )
    #     ),
    #     best_model=dict(
    #         stats=best_model_stats,
    #         plot_kwargs=dict(
    #             color='fuchsia',
    #             linestyle='-',
    #         )
    #     ),
    #     ensemble=dict(
    #         stats=best_ensemble_stats,
    #         plot_kwargs=dict(
    #             color='cyan',
    #             linestyle='-',
    #         )
    #     ),
    # )
    # plot_multi_auc(plot_data, subset, calibrator)

    #####################################            UQ           #####################################
    print("\nUQ")

    alea, epis, tota = compute_uq(y_pred_models, weights=gammas)
    # var, lin = compute_uq_comp(y_pred_models, weights=gammas)
    avrg_alea = []
    avrg_epis = []
    avrg_tota = []
    for a, e, t in zip(alea, epis, tota):
        avrg_alea.append(np.mean(a))
        avrg_epis.append(np.mean(e))
        avrg_tota.append(np.mean(t))
    avrg_alea = np.mean(avrg_alea)
    avrg_epis = np.mean(avrg_epis)
    avrg_tota = np.mean(avrg_tota)
    print(f"computing uq of ensemble models on {subset}..")
    print(f"avrg_alea: {avrg_alea:.3f}")
    print(f"avrg_epis: {avrg_epis:.3f}")
    print(f"avrg_tota: {avrg_tota:.3f}")


    for i in range(min(100, len(y_true))):
        # plot_pred_uq_comp_shot(y_pred_best_ensemble, y_true, var, lin, subset, calibrator, i)
        plot_pred_uq_shot(y_pred_best_ensemble, y_true, alea, epis, tota, subset, calibrator, i)