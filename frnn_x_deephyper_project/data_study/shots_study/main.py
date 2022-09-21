import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plasma.preprocessor.normalize import VarNormalizer as Normalizer

width = 7
height = width * 1.618

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

from plasma.conf_parser import parameters
from plasma.preprocessor.preprocess import guarantee_preprocessed

conf = parameters()
normalizer = Normalizer(conf)
normalizer.train()


signals_info = {
    0: dict(
        label="q95 safety factor",
        short="q95",
    ),
    1: dict(
        label="internal inductance",
        short="li",
        plot=dict(
            label="$I_i$",
            color="purple",
        ),
    ),
    2: dict(
        label="plasma current",
        short="ip",
        plot=dict(
            label="$I_p$",
            color="yellow",
        ),
    ),
    3: dict(
        label="Normalized Beta", 
        short="betan",
        plot=dict(
            label="$\\beta$",
            color="orangered",
        ),
    ),
    4: dict(
        label="stored energy",
        short="energy",
    ),
    5: dict(
        label="Locked mode amplitude",
        short="lm",
        plot=dict(
            label="$LM$",
            color="orangered",
        ),
    ),
    6: dict(
        label="Plasma density",
        short="dens",
        plot=dict(
            label="$n_e$",
            color="yellow",
        ),
    ),
    7: dict(
        label="Radiated Power Core",
        short="pradcore",
        plot=dict(
            label="$P_{rad,core}$",
            color="purple",
        ),
    ),
    8: dict(
        label="Radiated Power Edge",
        short="pradedge",
    ),
    9: dict(
        label="Input Power (beam for d3d)",
        short="pin",
        plot=dict(
            label="$P_{in}$",
            color="black",
        ),
    ),
    10: dict(
        label="Input Beam Torque",
        short="torquein",
    ),
    11: dict(
        label="plasma current direction",
        short="ipdirect",
    ),
    12: dict(
        label="plasma current target",
        short="iptarget",
        plot=dict(
            label="$I_{p,target}$",
            color="black",
        ),
    ),
    13: dict(
        label="plasma current error",
        short="iperr",
    ),
    14: dict(
        label="Electron temperature profile",
        short="etemp_profile",
        plot_label="$T_e(p)$"
    ),
    15: dict(
        label="Electron density profile",
        short="edens_profile",
        plot_label="$n_e(p)$"
    ),
}

group_views = {
    1: dict(
        sig_nums=[12, 1, 5, 2],
        lim=10,
    ),
    2: dict(
        sig_nums=[9, 7, 3, 6],
        lim=15,
    )
}

profile_views = {
    1: dict(
        sig_num=14,
        lim=10,
    ),
    2: dict(
        sig_num=15,
        lim=15,
    )
}

def plot_signal(ax, signals, sig_num):
    plot_kwargs = signals_info[sig_num]['plot']
    sig = signals[sig_num]
    # sig = shot.signals[sig_num]
    ax.plot(sig, color=plot_kwargs['color'], label=plot_kwargs['label'])

def plot_signal_group(ax, signals, sig_nums, lim, idx=None, disr=False):
    for sig_num in sig_nums:
        plot_signal(ax, signals, sig_num)
    ax.set_ylim([0, lim])
    title = f"#{idx}{' *' if disr else ''}" if idx is not None else None
    ax.legend(loc='upper left', title=title)

def plot_signal_profile(ax, signals, sig_num):
    sig = signals[sig_num]
    ax.imshow(np.flip(sig.transpose(), axis=0), cmap='inferno', aspect='auto')
    ax.set_ylabel(signals_info[sig_num]['plot_label'])
    # plt.colorbar(im, label=signals_info[sig_num]['plot_label'])

def plot_shot(shot, idx, suffix="png"):
    shot.restore(conf['paths']['processed_prepath'])
    disr = shot.is_disruptive
    normalizer.apply(shot)
    signals = list(shot.signals_dict.values())
    plt.figure()
    fig, ax = plt.subplots(len(group_views)+len(profile_views), sharex=True)
    for i, view in enumerate(group_views.values()):
        plot_signal_group(ax[i], signals, view['sig_nums'], view['lim'], idx=idx if i == 0 else None, disr=disr)
    for i, view in enumerate(profile_views.values()):
        plot_signal_profile(ax[i+len(group_views)], signals, view['sig_num'])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/shot_{idx}.{suffix}")
    plt.clf()

def plot_shot_separated(shot, idx, suffix="png"):
    shot.restore(conf['paths']['processed_prepath'])
    normalizer.apply(shot)
    signals = list(shot.signals_dict.values())
    plt.figure()
    fig, ax = plt.subplots(len(profile_views), sharex=True)
    for i, view in enumerate(profile_views.values()):
        plot_signal_profile(ax[i], signals, view['sig_num'])
    plt.tight_layout()
    plt.savefig(f"plots/profiles/shot_{idx}.{suffix}")
    plt.clf()
    fig, ax = plt.subplots(len(group_views), sharex=True)
    ax[0].set_title(f"test - shot_{idx}")
    for i, view in enumerate(group_views.values()):
        plot_signal_group(ax[i], signals, view['sig_nums'], view['lim'])
    plt.xlabel("timestep (in ms.)")
    plt.tight_layout()
    plt.savefig(f"plots/scalars/shot_{idx}.{suffix}")
    plt.clf()

shot_list_train, shot_list_valid, shot_list_test = guarantee_preprocessed(conf)
shot_list_test.sort()

for i in range(min(100, len(shot_list_test))):
    plot_shot_separated(shot_list_test[i], i, suffix="png")
    plot_shot(shot_list_test[i], i, suffix="png")
    plot_shot(shot_list_test[i], i, suffix="pdf")