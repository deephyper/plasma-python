import numpy as np
import yaml


def load_data(dataset, subset, model, group=None):
    if group is None:
        group = ""
    else:
        group = f"{group}/"
    data = np.load(f"predictions/{dataset}/{subset}/{group}{model}.npz")
    return list(data.values())


def load_pred_models(dataset, subset, group=None, criteria=None, k=1):
    if criteria is not None:
        with open(f"predictions/{dataset}/{criteria}/{group}/specs.yaml", 'r') as f:
            specs = yaml.safe_load(f)
        ordered_list = list(reversed(sorted([(specs[key]['auc'], key) for key in specs.keys()])))
    else:
        with open(f"predictions/{dataset}/valid/{group}/specs.yaml", 'r') as f:
            specs = yaml.safe_load(f)
        ordered_list = list(sorted([(int(key.split('_')[1]), key) for key in specs.keys()]))
    y_true = load_data(dataset, subset, "y_gold")
    pred_list = [[] for _ in range(len(y_true))]
    models = []
    if k is not None:
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


def average_prediction(y_pred_models, idx=None, weights=None):
    avg_y_pred = []
    if idx is None:
        for y_pred in y_pred_models:
            avg_y_pred.append(np.average(y_pred, weights=weights, axis=0))
    else:
        for y_pred in y_pred_models:
            avg_y_pred.append(np.average(y_pred[idx, ...], weights=weights, axis=0))
    return avg_y_pred

