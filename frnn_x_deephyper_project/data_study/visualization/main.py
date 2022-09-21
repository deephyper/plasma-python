import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP   
import matplotlib.pyplot as plt

from plasma.conf_parser import parameters
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader
from plasma.preprocessor.normalize import VarNormalizer as Normalizer
from plasma.utils.performance import PerformanceAnalyzer


# shot by shot (with 0 padding)
def shot_by_shot(signals, shot_lengths):
    max_shot_len = max(shot_lengths)
    n_features = signals[0].shape[1]

    resized_shots = []
    for signal, shot_length in zip(signals, shot_lengths):
        zero = np.zeros((max_shot_len*n_features))
        zero[:shot_length*n_features] = signal.flatten()
        resized_shots.append(zero)
    
    X = np.stack(resized_shots)
    return X


def dummy_shot_by_shot(signals, shot_lengths):
    max_shot_len = max(shot_lengths)
    n_features = signals[0].shape[1]

    resized_shots = []
    for shot_length in shot_lengths:
        zero = np.zeros((max_shot_len*n_features))
        zero[:shot_length*n_features] = 1
        resized_shots.append(zero)
    
    X = np.stack(resized_shots)
    return X

conf = parameters()

shot_list_train, shot_list_valid, shot_list_test = guarantee_preprocessed(conf)
normalizer = Normalizer(conf)
normalizer.train()
loader = Loader(conf, normalizer)
loader.verbose = False
signals_train, results_train, shot_lengths_train, disruptive_train = loader.get_signals_results_from_shotlist(shot_list_train, prediction_mode=True)
signals_valid, results_valid, shot_lengths_valid, disruptive_valid = loader.get_signals_results_from_shotlist(shot_list_valid, prediction_mode=True)
signals_test, results_test, shot_lengths_test, disruptive_test = loader.get_signals_results_from_shotlist(shot_list_test, prediction_mode=True)
print("Data OK")

print("Train")
print(f"N: {len(shot_lengths_train)}")
print(f"N_disr: {sum([1 if disr else 0 for disr in disruptive_train])}")
print(f"N_non_disr: {sum([0 if disr else 1 for disr in disruptive_train])}")
print(f"total_length: {sum(shot_lengths_train)}")

print("Valid")
print(f"N: {len(shot_lengths_valid)}")
print(f"N_disr: {sum([1 if disr else 0 for disr in disruptive_valid])}")
print(f"N_non_disr: {sum([0 if disr else 1 for disr in disruptive_valid])}")
print(f"total_length: {sum(shot_lengths_valid)}")

print("Test")
print(f"N: {len(shot_lengths_test)}")
print(f"N_disr: {sum([1 if disr else 0 for disr in disruptive_test])}")
print(f"N_non_disr: {sum([0 if disr else 1 for disr in disruptive_test])}")
print(f"total_length: {sum(shot_lengths_test)}")

signals, results, shot_lengths, disruptive = signals_test, results_test, shot_lengths_test, disruptive_test

X_generator = "shot_by_shot"
X_embedder = "TSNE"

if X_generator == "shot_by_shot":
    X = shot_by_shot(signals, shot_lengths)
else:
    X = dummy_shot_by_shot(signals, shot_lengths)
print("X OK")

if X_embedder == "TSNE":
    # embedder = TSNE()
    # X_embedded = embedder.fit_transform(X)
    X_embedded = np.load(f"embeddings/{X_embedder}.npy")
else:
    # embedder = UMAP()
    # X_embedded = embedder.fit_transform(X)
    X_embedded = np.load(f"embeddings/{X_embedder}.npy")
print(f"{X_embedder} OK")


y_pred = list((np.load("ensemble/y_pred.npz")).values())
y_true = list((np.load("ensemble/y_true.npz")).values())

analyser = PerformanceAnalyzer(conf=conf)
thresh_range, accuracy_range, precision_range, tp_rate_range, fp_rate_range, tp_range, fp_range = analyser.get_metrics_vs_p_thresh_dh(y_pred, y_true)

tp_fp_rate_range = tp_rate_range - fp_rate_range
id_max = np.argmax(tp_fp_rate_range)
thresh = thresh_range[id_max]
print(thresh)

ale_uq = list((np.load("ensemble/uq/alea.npz")).values())
epi_uq = list((np.load("ensemble/uq/epis.npz")).values())
tot_uq = list((np.load("ensemble/uq/tota.npz")).values())

pred_idx = np.array([np.argmax(pred[:-30]) if d else np.argmax(pred[:-30]) for pred, d in zip(y_pred, disruptive)])
y_pred = np.array([pred[pred_i] for pred, pred_i in zip(y_pred, pred_idx)]).flatten()

c_true = disruptive
c_stat = (y_pred>thresh).astype(int)+2*np.array(c_true) # 0: TN, 1: FP, 2: FN, 3: TP

y_ale_uq = np.array([ale[pred_i] for ale, pred_i in zip(ale_uq, pred_idx)]).flatten()/2.5
y_epi_uq = np.array([epi[pred_i] for epi, pred_i in zip(epi_uq, pred_idx)]).flatten()/2.5
y_tot_uq = np.array([tot[pred_i] for tot, pred_i in zip(tot_uq, pred_idx)]).flatten()/2.5
y_prop_uq = y_ale_uq/y_tot_uq

classes = dict(
    c_true=c_true,
    c_stat=c_stat,
)

disruptive_colors = {
    0: 'cyan',
    1: 'fuchsia',
}
stat_colors = {
    0: 'cyan', #TN
    1: 'fuchsia', #FP
    2: 'fuchsia', #FN
    3: 'cyan', #TP
}
c_colors = dict(
    c_true=disruptive_colors,
    c_stat=stat_colors,
)

y_reals = dict(
    y_ale_uq=y_ale_uq,
    y_epi_uq=y_epi_uq,
    y_tot_uq=y_tot_uq,
    y_prop_uq=y_prop_uq,
)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=[stat_colors[b] for b in c_stat], alpha=0.3, edgecolors=None)
plt.tight_layout()
print(f"saving {X_embedder}/thresh.png")
plt.savefig(f"{X_embedder}/thresh.png")

# for label, c in classes.items():
#     colors = c_colors[label]
#     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=[colors[b] for b in c], alpha=0.3, edgecolors=None)
#     plt.tight_layout()
#     print(f"saving {X_embedder}/{label}.png")
#     plt.savefig(f"{X_embedder}/{label}.png")

# for label, y in y_reals.items():
#     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, alpha=0.3, cmap='cool_r', edgecolors=None)
#     plt.tight_layout()
#     print(f"saving {X_embedder}/{label}.png")
#     plt.savefig(f"{X_embedder}/{label}.png")