# FRNN X DeepHyper Project

## Installation

To install everything that is needed, you can perform ``scalable_bo``'s installation process on *ThetaGPU* that you can find [here](https://github.com/deephyper/scalable-bo/blob/main/README.md#for-thetagpu-alcf). This will build the ``conda`` environment with all the ``plasma`` dependencies and the pipeline to run HPO with the ``minimalistic_frnn`` as problem (this problem is defined in ``scalbo/benchmarks/minimalistic_frnn.py`` of which you can find a copy in the ``hps/`` folder).

## Organization

Each part of this project is in a different folder, we have :

- ``data_study/`` : for generating plots to visualize the inputs.
- ``hps/`` : to run the HPO.
- ``training/`` : to select the models to train from the HPO' results and train them.
- ``prediction/`` : to perform trained model's predictions as well as ensemble construction, prediction, and uncertainty quantification.

Each of these is meant to be an experiment space, which is why you can find a ``jobs/`` folder (or one in each of their subfolders). For every submission script, be careful to modify these lines accordingly (maybe can this be automated with global variables)

```bash
#COBALT -O jobs/logs/main

PROJECT=/home/jgouneau/projects/grand/deephyper/frnn
source $PROJECT/repos/scalable-bo/build/activate-dhenv.sh

export PYTHONPATH=$PROJECT/repos/scalable-bo/build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

```

The ``#COBALT -O`` specifies the file's relative path in which are written the job's logs, be careful to create the according folders if any are in the path (for example here the folder ``logs/`` that I always create in ``jobs/``). The two other commands (``source`` and ``export``) refer to the ``conda`` environment build's location once it was created following the installation procedure. 

## Data Study

We have two subfolders in here ; ``visualization`` and ``shots_study``.

``visualization`` contains a script to generate UMAP and TSNE visualization of the datasets, but it wasn't really usefull. It uses the ground truth or the output of an ensemble's predictions (``ensemble/y_pred.npz``) or its results (TP, FP, etc. with a specified threshold) or its (``ensemble/uq/alea.npz``, ``ensemble/uq/epis.npz``, ``ensemble/uq/tota.npz``) to color the points. It wasn't a successful attempt so it is still in a state of raw code uneasy to use for which it would be a loss of time to enter the details.

``shots_study`` is where we generate the inputs visualizations. First create a folder ``plots/`` in here, in which you can also create ``scalars`` and ``profiles``, everything happens at the end of the script (``l.210 -> l.216``) :

```python
shot_list = shot_list_test
shot_list.sort()
end = 100
for i in range(min(end, len(shot_list))):
    plot_shot_separated(shot_list[i], i, suffix="png")
    plot_shot(shot_list[i], i, suffix="png")
    plot_shot(shot_list[i], i, suffix="pdf")
```

You can select the dataset from which you want to generate the inputs visualizations with ``shot_list = shot_list_test`` as well as until which one with ``end = 100``. Then you can choose which views are generated ; 
- ``plot_shot()`` creates the whole view (with scalars and profiles in the same figure) directly in the ``plots/`` folder
- ``plot_shot_separated()`` creates the scalars and profiles views separatley in the corresponding folders ``plots/scalars/`` and ``plots/profiles/`` (this was usefull for the shared Google Sheet)
You can choose in which format to save it with the ``suffix`` argument.

To run this script, execute ``qsub-gpu jobs/main.sh`` in the current folder.

## HPS

## Training

## Prediction

