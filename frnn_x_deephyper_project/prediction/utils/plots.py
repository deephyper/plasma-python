import matplotlib
import matplotlib.pyplot as plt

import numpy as np

width = 7
height = width / 1.618

matplotlib.rcParams.update({
    'font.size': 12,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 75,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})


def plot_multi_roc(plot_data, save_label):
    plt.figure()
    for data in plot_data.values():
        stats = data['stats']
        plot_kwargs = data['plot_kwargs']
        tp_rate_range = stats['tp_rate_range']
        fp_rate_range = stats['fp_rate_range']
        plt.plot(fp_rate_range, tp_rate_range, **plot_kwargs)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.legend(list(plot_data.keys()))
    plt.ylabel("True Positives Ratio")
    plt.xlabel("False Positives Ratio")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/roc/{save_label}.png")
    plt.clf()


def plot_roc_curves(baseline_stats, best_model_stats, ensemble_stats, save_label):
    plot_data = {
        "baseline": dict(
            stats=baseline_stats,
            plot_kwargs=dict(
                color='navy',
                linestyle='-',
            )
        ),
        "best model": dict(
            stats=best_model_stats,
            plot_kwargs=dict(
                color='fuchsia',
                linestyle='-',
            )
        ),
        # ensemble=dict(
        #     stats=ensemble_stats,
        #     plot_kwargs=dict(
        #         color='cyan',
        #         linestyle='-',
        #     )
        # ),
    }
    plot_multi_roc(plot_data, save_label)


def plot_pred_shot(ax, y_pred, y_true, ylabel="$y_\\mathcal{E}$", title=None):
    true_color = 'navy'
    # alarm_time = get_alarm_time(y_pred)
    shot_length = len(y_true)

    ax.set_title(title)
    ax.plot(y_true, color=true_color)
    ax.plot(y_pred, color='black', label="$\\hat{y}$")
    ax.fill_between(list(range(shot_length)), y_true.flatten(), 0.5, color=true_color, alpha=0.5, label="$y$")
    # if alarm_time is not None:
    #     ax.plot(alarm_time, 0.02, marker='^', color='black', markersize=10, zorder=100)
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length-1])
    ax.grid(linestyle="--")
    ax.legend(loc='upper left')


def plot_pred_separate_shot(ax, y_pred_models, y_true, weights=None):
    true_color = 'navy'
    # alarm_time = get_alarm_time(y_pred)
    shot_length = len(y_true)

    if weights is not None:
        weights = np.array(weights)
        a = weights.min()
        b = weights.max()
        def range_norm(val, a=0, b=1, c=0, d=1):
            val = (val-a)*(d-c)/(b-a)+c
            return val
        colors = plt.cm.gnuplot2(range_norm(weights, a=a, b=b, c=0.2, d=0.7))
    else:
        colors = None

    ax.plot(y_true, color=true_color)
    for i in range(y_pred_models.shape[0]):
        ax.plot(y_pred_models[i, ...], color='black', alpha=weights[i]*2 if weights is not None else 0.5, label="$\\hat{y}$" if i == 0 else None)
    ax.fill_between(list(range(shot_length)), y_true.flatten(), 0.5, color=true_color, alpha=0.5, label="$y$")
    # if alarm_time is not None:
    #     ax.plot(alarm_time, 0.02, marker='^', color='black', markersize=10, zorder=100)
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.set_ylabel("$y_\\mathcal{E}$")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length-1])
    ax.grid(linestyle="--")
    ax.legend(loc='upper left')


def plot_uq_shot(ax, var_ale, var_epi, var_tot, idx=None):
    if idx is not None:
        idx = f"#{idx}"
    shot_length = len(var_tot)
    ax.plot(var_tot, color='black')
    ax.fill_between(list(range(shot_length)), var_epi.flatten(), color='fuchsia', alpha=0.5)
    ax.fill_between(list(range(shot_length)), (var_epi+var_ale).flatten(), var_epi.flatten(), color='cyan', alpha=0.5)
    ax.set_ylabel("$u_{\\mathcal{E}, norm}$")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length-1])
    ax.grid(linestyle="--")
    ax.legend(["$u_{tot}$", "$u_e$", "$u_a$"], loc='upper left', title=idx)


def plot_uq_shot_single_model(ax, u, idx=None):
    if idx is not None:
        idx = f"#{idx}"
    shot_length = len(u)
    ax.plot(u, color='black')
    ax.fill_between(list(range(shot_length)), u.flatten(), color='cyan', alpha=0.5)
    ax.set_ylabel("$u_{\\theta, norm}$")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, shot_length-1])
    ax.grid(linestyle="--")
    ax.legend(["$u$"], loc='upper left', title=idx)


def plot_pred_uq_shot(y_pred_list, y_true_list, var_ale_list, var_epi_list, var_tot_list, subset, save_dir, idx, suffix="png"):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    plot_uq_shot(ax[0], var_ale_list[idx], var_epi_list[idx], var_tot_list[idx], idx=idx)
    # ax[0].set_title(f"{subset} - shot {idx}")
    plot_pred_shot(ax[1], y_pred_list[idx], y_true_list[idx])
    plt.xlabel("Timestep (ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/uq/{save_dir}/shot_{idx}.{suffix}")
    plt.clf()


def plot_pred_uq_shot_separate(y_pred_models_list, y_true_list, var_ale_list, var_epi_list, var_tot_list, subset, save_dir, idx, weights=None, suffix="png"):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    plot_uq_shot(ax[0], var_ale_list[idx], var_epi_list[idx], var_tot_list[idx], idx=idx)
    # ax[0].set_title(f"{subset} - shot {idx}")
    plot_pred_separate_shot(ax[1], y_pred_models_list[idx], y_true_list[idx], weights=weights)
    plt.xlabel("Timestep (ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/uq/{save_dir}/shot_{idx}.{suffix}")
    plt.clf()


def plot_pred_uq_shot_single_model(y_pred_list, y_true_list, u, subset, save_dir, idx, suffix="png"):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    plot_uq_shot_single_model(ax[0], u[idx], idx=idx)
    plot_pred_shot(ax[1], y_pred_list[idx], y_true_list[idx], ylabel="$y_\\theta$")
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/uq/{save_dir}/shot_{idx}.{suffix}")
    plt.clf()