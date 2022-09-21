import numpy as np

from plasma.utils.performance import PerformanceAnalyzer
analyser = PerformanceAnalyzer(T_min_warn=30 , T_max_warn=1024, ignore_timesteps=100)



def get_stats(y_true, y_pred):
    thresh_range, accuracy_range, precision_range, tp_rate_range, fp_rate_range, tp_range, fp_range, tn_range, fn_range = analyser.get_metrics_vs_p_thresh_dh(y_pred, y_true, early_pred_counts=True)
    stats = dict(
        thresh_range=thresh_range,
        accuracy_range=accuracy_range,
        precision_range=precision_range,
        tp_rate_range=tp_rate_range,
        fp_rate_range=fp_rate_range,
        tp_range=tp_range,
        fp_range=fp_range,
        tn_range=tn_range,
        fn_range=fn_range,
    )
    return stats


def get_thresh_stats(stats, thresh):
    thresh_range = stats['thresh_range']
    idx = np.argmin(np.abs(thresh_range-thresh))
    if stats['thresh_range'][idx] > thresh and idx > 0:
        idx -= 1
    thresh = stats['thresh_range'][idx]
    accuracy = stats['accuracy_range'][idx]
    precision = stats['precision_range'][idx]
    tp_rate = stats['tp_rate_range'][idx]
    fp_rate = stats['fp_rate_range'][idx]
    tp = stats['tp_range'][idx]
    fp = stats['fp_range'][idx]
    tn = stats['tn_range'][idx]
    fn = stats['fn_range'][idx]
    thresh_stats = dict(
        thresh=thresh,
        accuracy=accuracy,
        precision=precision,
        tp_rate=tp_rate,
        fp_rate=fp_rate,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
    return thresh_stats



def compute_auc_from_stats(stats):
    tp_rate_range = stats['tp_rate_range']
    fp_rate_range = stats['fp_rate_range']
    auc = analyser.roc_from_tp_fp_rates(tp_rate_range, fp_rate_range)
    return auc


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


def compute_uq_single_model(y_pred):
    u_list = []
    for p in y_pred:
        u = np.minimum(p, 1-p)
        u_list.append(u/0.5)
    return u_list