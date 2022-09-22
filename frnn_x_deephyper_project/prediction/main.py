import time
import numpy as np
import copy

from utils.algorithms import apply_calibration_model, apply_balanced_calibration_model, apply_calibration_models, apply_balanced_calibration_models, apply_caruana, apply_calibration_ensemble, loss_balanced_crossentropy, loss_auc
from utils.managing import load_data, load_pred_models, average_prediction
from utils.evaluation import get_stats, get_thresh_stats, compute_auc_from_stats, compute_uq, compute_uq_single_model
from utils.plots import plot_roc_curves, plot_pred_uq_shot, plot_pred_uq_shot_separate, plot_pred_uq_shot_single_model

if __name__ == "__main__":
    dataset = "d3d_2019"
    subset = "test"
    criteria = "valid"
    group = "top_models"
    method = "gradient" # "new_method", "caruana", "topk"
    top = 80
    methods_kwargs = dict(
        topk=dict(
            calibrator="balanced_sigmoid", # "base", "sigmoid", "balanced_sigmoid"
            k=80,
        ),
        caruana=dict(
            calibrator="balanced_sigmoid", # "base", "sigmoid", "balanced_sigmoid"
            k=80,
        ),
        gradient=dict(
            ensemble=list(range(top)),
            keep_model_thresh=1e-06,
        ),
    )


    method_kwargs = methods_kwargs[method]
    print(f"\ndataset: {dataset} - subset: {subset} - criteria: {criteria}")
    print(f"group: {group} - method: {method} - kwargs: {method_kwargs}")

    # LOAD PREDICTIONS
    # baseline, ensemble(K, order), best_model, y_true
    y_true = load_data(dataset, subset, "y_gold")
    y_pred_baseline = load_data(dataset, subset, "baseline")
    y_pred_models, models = load_pred_models(dataset, subset, group=group, criteria=criteria, k=top)
    y_pred_best_model = copy.deepcopy(average_prediction(y_pred_models, idx=[0]))
    # print(f"models: {models}")

    # APPLY CALIBRATION/ENSEMBLE
    print("Executing the Method")
    T = 0
    weights = None
    for i, pred in enumerate(y_pred_baseline):
        y_pred_baseline[i] = 1/(1+np.exp(-pred))
    if method in ["topk", "caruana"]:
        calibrator = method_kwargs['calibrator']
        k = method_kwargs['k']
        t = time.time()
        if calibrator == "sigmoid":
            # y_pred_baseline = apply_calibration_model(y_pred_baseline, y_true, dataset, subset, criteria, "baseline")
            y_pred_best_model = apply_calibration_model(y_pred_best_model, y_true, dataset, subset, criteria, models[0], group=group)
            t = time.time()
            y_pred_models = apply_calibration_models(y_pred_models, y_true, dataset, subset, criteria, group, models)
            T += time.time() - t
        elif calibrator == "balanced_sigmoid":
            # y_pred_baseline = apply_balanced_calibration_model(y_pred_baseline, y_true, dataset, subset, criteria, "baseline")
            y_pred_best_model = apply_balanced_calibration_model(y_pred_best_model, y_true, dataset, subset, criteria, models[0], group=group)
            t = time.time()
            y_pred_models = apply_balanced_calibration_models(y_pred_models, y_true, dataset, subset, criteria, group, models)
            T += time.time() - t
        else: # calibrator == "base"
            # for i, pred in enumerate(y_pred_baseline):
            #     y_pred_baseline[i] = 1/(1+np.exp(-pred))
            for i, pred in enumerate(y_pred_best_model):
                y_pred_best_model[i] = 1/(1+np.exp(-pred))
            t = time.time()
            for i, pred in enumerate(y_pred_models):
                y_pred_models[i] = 1/(1+np.exp(-pred))
            T += time.time() - t
        if method == "caruana":
            t = time.time()
            ensemble = apply_caruana(loss_balanced_crossentropy, y_pred_models, y_true, k, dataset, subset, criteria, calibrator, group)
            T += time.time() - t
        else:
            t = time.time()
            ensemble = list(range(k))
            T += time.time() - t
    else: # method == "gradient":
        ensemble = method_kwargs['ensemble']
        keep_model_thresh = method_kwargs['keep_model_thresh']
        # y_pred_baseline = apply_balanced_calibration_model(y_pred_baseline, y_true, dataset, subset, criteria, "baseline")
        y_pred_best_model = apply_balanced_calibration_model(y_pred_best_model, y_true, dataset, subset, criteria, models[0], group=group)
        t = time.time()
        y_pred_models, weights, ensemble = apply_calibration_ensemble(y_pred_models, y_true, dataset, subset, criteria, group, keep_models_thresh=keep_model_thresh)
        T += time.time() - t
    t = time.time()
    y_pred_models_selection = []
    for y_pred in y_pred_models:
        y_pred_models_selection.append(y_pred[ensemble, ...])
    y_pred_ensemble = average_prediction(y_pred_models_selection, weights=weights)
    T += time.time() - t
    print(f"Execution took {T:.2f}s.")

    # EVALUATE
    # baseline, best_model, ensemble
    baseline_stats = get_stats(y_true, y_pred_baseline)
    best_model_stats = get_stats(y_true, y_pred_best_model)
    ensemble_stats = get_stats(y_true, y_pred_ensemble)

    baseline_thresh_stats = get_thresh_stats(baseline_stats, 0.5)
    best_model_thresh_stats = get_thresh_stats(best_model_stats, 0.5)
    ensemble_thresh_stats = get_thresh_stats(ensemble_stats, 0.5)

    # plot_roc_curves(baseline_stats, best_model_stats, ensemble_stats, "roc_curve_test")

    print("\nSTATS")
    print("\nbaseline")
    for key, val in baseline_thresh_stats.items():
        print(f"{key}: {val}")
    print("\nbest_model")
    for key, val in best_model_thresh_stats.items():
        print(f"{key}: {val}")
    print("\nensemble")
    for key, val in ensemble_thresh_stats.items():
        print(f"{key}: {val}")

    print("\nAUC")
    baseline_auc = compute_auc_from_stats(baseline_stats)
    best_model_auc = compute_auc_from_stats(best_model_stats)
    best_ensemble_auc = compute_auc_from_stats(ensemble_stats)
    print(f"baseline_auc: {baseline_auc}")
    print(f"best_model_auc: {best_model_auc}")
    print(f"best_ensemble_auc: {best_ensemble_auc}")

    print("\nBalanced CE")
    baseline_balanced_crossentropy = loss_balanced_crossentropy(y_true, y_pred_baseline)
    best_model_balanced_crossentropy = loss_balanced_crossentropy(y_true, y_pred_best_model)
    ensemble_balanced_crossentropy = loss_balanced_crossentropy(y_true, y_pred_ensemble)
    print(f"baseline_balanced_CE: {baseline_balanced_crossentropy}")
    print(f"best_model_balanced_CE: {best_model_balanced_crossentropy}")
    print(f"ensemble_balanced_CE: {ensemble_balanced_crossentropy}")


    # UQ
    # baseline
    u_base = compute_uq_single_model(y_pred_baseline)
    u_best = compute_uq_single_model(y_pred_best_model)
    # ensemble
    print("\nUQ")
    alea, epis, tota = compute_uq(y_pred_models_selection, weights=weights)
    avrg_alea = np.mean(np.concatenate(alea))
    avrg_epis = np.mean(np.concatenate(epis))
    avrg_tota = np.mean(np.concatenate(tota))
    print(f"avrg_alea: {avrg_alea}")
    print(f"avrg_epis: {avrg_epis}")
    print(f"avrg_tota: {avrg_tota}")

    print("\nPLOTS")
    plot_roc_curves(baseline_stats, best_model_stats, ensemble_stats, "roc_curve")

    num_shots = 100
    for i in range(min(num_shots, len(y_true))):
        plot_pred_uq_shot_single_model(y_pred_baseline, y_true, u_base, subset, f"{subset}/baseline", i, suffix="png")
        plot_pred_uq_shot_single_model(y_pred_best_model, y_true, u_best, subset, f"{subset}/best_model", i, suffix="png")

    if method in ["topk", "caruana"]:
        calibrator = method_kwargs['calibrator'] if method_kwargs['calibrator'] in ["sigmoid", "balanced_sigmoid"] else "no_calibration"
        for i in range(min(num_shots, len(y_true))):
            plot_pred_uq_shot(y_pred_ensemble, y_true, alea, epis, tota, subset, f"{subset}/{group}/{method}/{calibrator}", i, suffix="png")
            # plot_pred_uq_shot_separate(y_pred_models_selection, y_true, alea, epis, tota, subset, f"{subset}/{group}/{method}/{calibrator}/separated", i, suffix="png")
    else:
        for i in range(min(num_shots, len(y_true))):
            plot_pred_uq_shot(y_pred_ensemble, y_true, alea, epis, tota, subset, f"{subset}/{group}/{method}", i, suffix="png")
            # plot_pred_uq_shot_separate(y_pred_models_selection, y_true, alea, epis, tota, subset, f"{subset}/{group}/{method}/separated", i, weights=weights, suffix="png")
    