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

The ``hp_problem`` is defined in the ``minimalistic_frnn`` benchmark from ``scalbo`` (``l.80->l.96``) (not the one given as example in ``hps/minimalistic_frnn.py``):

```python
hp_problem = HpProblem()
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "batch_size", default_value=128)
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "dense_size", default_value=128)
hp_problem.add_hyperparameter((0.0, 1.0), "dense_regularization", default_value=0.001)
hp_problem.add_hyperparameter((0.0, 0.5), "dropout_prob", default_value=0.1)
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "length", default_value=128)
hp_problem.add_hyperparameter(['hinge', 'cross', 'focal', 'balanced_hinge', 'balanced_cross', 'balanced_focal'], "loss", default_value='focal')
hp_problem.add_hyperparameter((1e-7, 1e-2, "log-uniform"), "lr", default_value=2e-5)
hp_problem.add_hyperparameter((0.9, 1.0), "lr_decay", default_value=0.97)
hp_problem.add_hyperparameter((0.9, 1.0), "momentum", default_value=0.9)
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "num_conv_filters", default_value=128)
hp_problem.add_hyperparameter((1, 4), "num_conv_layers", default_value=3)
hp_problem.add_hyperparameter((1, 32, "log-uniform"), "num_epochs", default_value=32)
hp_problem.add_hyperparameter((0.0, 1.0), "regularization", default_value=0.001)
hp_problem.add_hyperparameter((1, 4), "rnn_layers", default_value=2)
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "rnn_size", default_value=200)
```

as well as the ``run()`` function (``l.541->l.630``)

The allocated time for the training of each model is defined in seconds ``l.585``:

```python
timeout_callback = TimeoutCallback(30*60)
```

While the parameters of the search are defined in the submission script (the final one is ``jobs/minimalistic-frnn-DBO-async-qUCB-qUCB-16-8-42000-42.sh``) with ``scalbo``'s cli : 

```bash
#COBALT -n 16
#COBALT -t 720

export RANKS_PER_NODE=8
export COBALT_JOBSIZE=16

export acq_func="qUCB"
export strategy="qUCB"
export timeout=42000
export random_state=42 
export problem="minimalistic-frnn"
export sync_val=0
export search="DBO"
```

``#COBALT -n 16`` and ``export COBALT_JOBSIZE=16`` must be the same, ``#COBALT -t 720`` is the maximum submission time of 12h, and ``export timeout=42000`` corresponds to a search of 11h40min.

The results are saved in ``export log_dir="results/$problem-$search-$sync_str-$acq_func-$strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state"`` so make sure that you have a ``results/`` folder in the current folder.

There are other scripts with different parameters, as well as scripts to quickly test the search on 1 gpu or 1 node (``test_HPS_1_gpu.sh``, ``test_HPS_1.sh``) or even only the run function with the baseline (``test_base_run.sh``, to be more precise it executes what's in the ``__main__`` part of ``scalbo``'s ``minimalistic_frnn.py`` benchmark script).

## Training

Once the HPO is done its results are in ``hps/results/minimalistic-frnn-DBO-async-qUCB-qUCB-16-8-42000-42/results.csv``.

First execute (this can be done localy) the ``gather_top_k_configs.py`` script for which the key variables are defined at the beginning (``l.4->l.8``):

```python
path_to_results = "../hps/results/minimalistic-frnn-DBO-async-qUCB-qUCB-16-8-42000-42/results.csv"
k = 80
path_to_configs = f"configs/top_{k}.json"
```

it will gather the top ``k`` results from ``path_to_results`` and save them in ``configs/top_'k'.json``  (in which we already have the baseline config).

To then perform the training of the top models gathered, it happens with the ``train_top_k.py`` script, which takes the top configs generated previously in ``path_to_top_configs`` an then simply reproduces the ``run()`` function with an added ``FrnnEvaluatorCallback`` whose role is to evaluate the model at each epoch, save the metrics evaluated in ``'results_path'/histories/'model_name'.json`` and save the model's weights in ``'results_path'/model_weights/'model_name'.h5`` if the ``valid_frnn_auc`` was improved. 

All the key variables are defined ``l.593->l.595`` :

```python
path_to_top_configs = "configs/top_80.json"
results_path = "results/top_80"
model_name = f"top_{rank+1}"
```

Make sure you have the ``results_path`` folder created in ``training/``. Also because of the way the ``FrnnEvaluatorCallback`` checkpoints the model, make sure you have both a ``histories/`` and a ``model_weights/`` folders in this ``results_path`` folder.

The number of epochs during which to run the training is defined ``l.555``:

```python
num_epochs = 128
```
with the periodic evaluation of the model it is not necessary for the training to be finished by the end of the 12h job so this can be set to a very large value.

To compare to the baseline it is also possible to run ``train_baseline.py`` (even though I left the results of this training in ``results/baseline/``) but it will take one node to train only one model, this can be improved. Also, just like for the top models' training, make sure you have the folders created for the ``FrnnEvaluatorCallback`` to checkpoint the baseline's training.

There are also two other scripts, which don't checkpoint the training and run for only an hour on ``single-gpu``, to compare the gpu utilization of the two buffer methods o the baseline (ours ``train_baseline_buffer.py`` and the original one ``train_baseline_old_buffer.py``) ; the ``gpustat`` outputs are written in ``results/gpustat_buffer.txt`` and ``results/gpustat_old_buffer.txt`` respectively so make sure you have a ``results/`` folder in ``training/``. To get the gpu utilization from the generated gpustats file you can use this snippet with the correct ``"path/to/gpustat.txt"`` :

```python
with open("path/to/gpustat.txt", 'r') as f:
    use = []
    for line in f:
        if line.startswith('['):
            use.append(int(line.split('%')[0].split(' ')[-2]))
print(sum(use)/len(use))
```

All these training scripts have their associated job submission script in ``jobs/``, as always.

## Prediction

