from __future__ import print_function
import plasma.global_vars as g
from plasma.primitives.shots import ShotListFiles
import data.signals as sig
from plasma.utils.hashing import myhash_signals
# from data.signals import (
#     all_signals, fully_defined_signals_1D,
#     jet, d3d)  # nstx
import os
import getpass
import copy

HERE = os.path.dirname(os.path.abspath(__file__))

base_params = {
    # 'fs_path': '/lus/grand/projects/datascience/jgouneau/deephyper/frnn/dataset',
    'fs_path': '/lus/theta-fs0/projects/fusiondl_aesp/felker',
    'user_subdir': False,
    'fs_path_output': '/lus/theta-fs0/projects/datascience/jgouneau/deephyper/frnn/scalable-bo/experiments/thetagpu/jobs/output/frnn/',
    'user_subdir_output': False,
    'target': 'hinge',
    'num_gpus': 1,
    'paths': {
        # 'signal_prepath': '/signal_data/',
        'signal_prepath': ['/signal_data/', '/signal_data_new_nov2019/'],
        'shot_list_dir': '/shot_lists/',
        'tensorboard_save_path': '/Graph/',
        # 'data': 'jet_0D',
        # 'data': 'd3d_all',
        'data': 'd3d_2019',
        'specific_signals': [],
        'executable': 'mpi_learn.py',
        'shallow_executable': 'learn.py',
    },
    'data': {
        'bleed_in': 0,
        'bleed_in_repeat_fac': 1,
        'bleed_in_remove_from_test': True,
        'bleed_in_equalize_sets': False,
        'signal_to_augment': 'None',
        'augmentation_mode': 'None',
        'augment_during_training': False,
        'cut_shot_ends': True,
        'recompute': False,
        'recompute_normalization': False,
        'current_index': 0,
        'plotting': False,
        'use_shots': 200000,
        'positive_example_penalty': 1.0,
        'dt': 0.001,
        'T_min_warn': 30,
        'T_max': 1000.0,
        'T_warning': 1.024,
        'current_thresh': 750000,
        'current_end_thresh': 10000,
        'window_decay': 2,
        'window_size': 10,
        'normalizer': 'var',
        'norm_stat_range': 100.0,
        'equalize_classes': False,
        'floatx': 'float32',
    },
    'model': {
        'loss_scale_factor': 1.0,
        'use_batch_norm': False,
        'torch': False,
        'shallow': False,
        'shallow_model': {
            'type': 'xgboost',
            'num_samples': 1000000,
            'n_estimators': 100,
            'max_depth': 3,
            'C': 1.0,
            'kernel': 'rbf',
            'learning_rate': 0.1,
            'scale_pos_weight': 10.0,
            'final_hidden_layer_size': 10,
            'num_hidden_layers': 3,
            'learning_rate_mlp': 0.0001,
            'mlp_regularization': 0.0001,
            'skip_train': False,
        },
        'pred_length': 128,
        'pred_batch_size': 128,
        'length': 128,
        'skip': 1,
        'rnn_size': 200,
        'rnn_type': 'CuDNNLSTM',
        'rnn_layers': 2,
        'num_conv_filters': 128,
        'size_conv_filters': 3,
        'num_conv_layers': 3,
        'pool_size': 2,
        'dense_size': 128,
        'extra_dense_input': False,
        'optimizer': 'adam',
        'clipnorm': 10.0,
        'regularization': 0.001,
        'dense_regularization': 0.001,
        'lr': 2e-05,
        'lr_decay': 0.97,
        'stateful': True,
        # 'stateful': False,
        'return_sequences': True,
        'dropout_prob': 0.1,
        'warmup_steps': 0,
        'ignore_timesteps': 100,
        'backend': 'tensorflow',
    },
    'training': {
        'as_array_of_shots': True,
        'shuffle_training': True,
        'train_frac': 0.75,
        'validation_frac': 0.3333333333333333,
        'batch_size': 128,
        'max_patch_length': 100000,
        'num_shots_at_once': 200,
        'num_epochs': 4,
        'use_mock_data': False,
        'data_parallel': False,
        'hyperparam_tuning': False,
        'batch_generator_warmup_steps': 0,
        'use_process_generator': False,
        'num_batches_minimum': 20,
        'ranking_difficulty_fac': 1.0,
        'timeline_prof': False,
        'step_limit': -1,
        'no_validation': False,
    },
    'callbacks': {
        'list': ['earlystop'],
        'metrics': ['val_loss',
        'val_roc',
        'train_loss'],
        'mode': 'max',
        'monitor': 'val_roc',
        'patience': 5,
        'write_grads': False,
        'monitor_test': True,
        'monitor_times': [30, 70, 200, 500, 1000],
    },
    'env': {
        'name': 'frnn',
        'type': 'anaconda',
    },
}

def modify_config(config):
    params = copy.deepcopy(base_params)
    if config is not None :
        fs_path = config.get("fs_path", None)
        if fs_path is not None:
            params['fs_path'] = fs_path
        params['model']['length'] = config['length']
        params['model']['pred_length'] = config['length']
        params['model']['rnn_size'] = config['rnn_size']
        params['model']['rnn_layers'] = config['rnn_layers']
        params['model']['num_conv_filters'] = config['num_conv_filters']
        params['model']['num_conv_layers'] = config['num_conv_layers']
        params['model']['dense_size'] = config['dense_size']
        params['model']['regularization'] = config['regularization']
        params['model']['dense_regularization'] = config['dense_regularization']
        params['model']['lr'] = config['lr']
        params['model']['lr_decay'] = config['lr_decay']
        params['model']["momentum"] = config['momentum']
        params['model']['dropout_prob'] = config['dropout_prob']
        params['training']['batch_size'] = config['batch_size']
        params['model']['pred_batch_size'] = config['batch_size']

    return params


def parameters(config=None):
    """Parse yaml file of configuration parameters."""
    # TODO(KGF): the following line imports TensorFlow as a Keras backend
    # by default (absent env variable KERAS_BACKEND and/or config file
    # $HOME/.keras/keras.json) "from plasma.conf import conf"
    # via "import keras.backend as K" in targets.py
    from plasma.models.targets import (
        HingeTarget, MaxHingeTarget, BinaryTarget,
        TTDTarget, TTDInvTarget, TTDLinearTarget
        )
    params = modify_config(config)
    params['user_name'] = getpass.getuser()
    base_path = params['fs_path']
    if params['user_subdir']:
        base_path = os.path.join(base_path, params['user_name'])
    output_path = params['fs_path_output']
    if params['user_subdir_output']:
        output_path = os.path.join(output_path, params['user_name'])

    params['paths']['base_path'] = base_path
    params['paths']['output_path'] = output_path
    if isinstance(params['paths']['signal_prepath'], list):
        g.print_unique('Reading from multiple data folders!')
        params['paths']['signal_prepath'] = [
            base_path + s for s in params['paths']['signal_prepath']]
    else:
        params['paths']['signal_prepath'] = (
            base_path + params['paths']['signal_prepath'])
    params['paths']['shot_list_dir'] = (
        base_path + params['paths']['shot_list_dir'])
    # See notes in data/signals.py for details on signal tols relative to
    # t_disrupt. The following 2x dataset definitions permit progressively
    # worse signal quality when preprocessing the shots and omitting some
    if params['paths']['data'] == 'd3d_data_max_tol':
        # let signals terminate up to 29 ms before t_disrupt on D3D
        h = myhash_signals(sig.all_signals_max_tol.values())
    elif params['paths']['data'] == 'd3d_data_garbage':
        # let up to 3x signals disappear at any time before t_disrupt
        # (and NaNs?)
        # -----
        # temp workaround for identical signal dictionary (but different
        # omit criteria in shots.py Shot.get_signals_and_times_from_file())
        # ---> 2x hash int
        # TODO(KGF): not robust; create reproducible specification and
        # recording of signal filtering procedure
        h = myhash_signals(sig.all_signals_max_tol.values())*2
    else:
        h = myhash_signals(sig.all_signals.values())

    params['paths']['global_normalizer_path'] = (
        base_path
        + '/normalization/normalization_signal_group_{}.npz'.format(h))
    if params['training']['hyperparam_tuning']:
        # params['paths']['saved_shotlist_path'] =
        # './normalization/shot_lists.npz'
        params['paths']['normalizer_path'] = (
            './normalization/normalization_signal_group_{}.npz'.format(h))
        params['paths']['model_save_path'] = './model_checkpoints/'
        params['paths']['csvlog_save_path'] = './csv_logs/'
        params['paths']['results_prepath'] = './results/'
    else:
        # params['paths']['saved_shotlist_path'] = output_path +
        # '/normalization/shot_lists.npz'
        params['paths']['normalizer_path'] = (
            params['paths']['global_normalizer_path'])
        params['paths']['model_save_path'] = (output_path
                                                + '/model_checkpoints/')
        params['paths']['csvlog_save_path'] = output_path + '/csv_logs/'
        params['paths']['results_prepath'] = output_path + '/results/'
    params['paths']['tensorboard_save_path'] = (
        output_path + params['paths']['tensorboard_save_path'])
    params['paths']['saved_shotlist_path'] = (
        params['paths']['base_path'] + '/processed_shotlists/'
        + params['paths']['data']
        + '/shot_lists_signal_group_{}.npz'.format(h))
    params['paths']['processed_prepath'] = (
        base_path + '/processed_shots/' + 'signal_group_{}/'.format(h))
    # ensure shallow model has +1 -1 target.
    if params['model']['shallow'] or params['target'] == 'hinge':
        params['data']['target'] = HingeTarget
    elif params['target'] == 'maxhinge':
        MaxHingeTarget.fac = params['data']['positive_example_penalty']
        params['data']['target'] = MaxHingeTarget
    elif params['target'] == 'binary':
        params['data']['target'] = BinaryTarget
    elif params['target'] == 'ttd':
        params['data']['target'] = TTDTarget
    elif params['target'] == 'ttdinv':
        params['data']['target'] = TTDInvTarget
    elif params['target'] == 'ttdlinear':
        params['data']['target'] = TTDLinearTarget
    else:
        # TODO(KGF): "Target" base class is unused here
        g.print_unique('Unknown type of target. Exiting')
        exit(1)

    # params['model']['output_activation'] =
    # params['data']['target'].activation
    # binary crossentropy performs slightly better?
    # params['model']['loss'] = params['data']['target'].loss

    # signals
    if params['paths']['data'] in ['d3d_data_max_tol', 'd3d_data_garbage']:
        params['paths']['all_signals_dict'] = sig.all_signals_max_tol
    else:
        params['paths']['all_signals_dict'] = sig.all_signals

    # assert order
    # q95, li, ip, lm, betan, energy, dens, pradcore, pradedge, pin,
    # pechin, torquein, ipdirect, etemp_profile, edens_profile

    # shot lists
    jet_carbon_wall = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['CWall_clear.txt', 'CFC_unint.txt'], 'jet carbon wall data')
    jet_iterlike_wall = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['ILW_unint.txt', 'BeWall_clear.txt'], 'jet iter like wall data')
    jet_iterlike_wall_late = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['ILW_unint_late.txt', 'ILW_clear_late.txt'],
        'Late jet iter like wall data')
    # jet_iterlike_wall_full = ShotListFiles(
    #     sig.jet, params['paths']['shot_list_dir'],
    #     ['ILW_unint_full.txt', 'ILW_clear_full.txt'],
    #     'Full jet iter like wall data')

    jenkins_jet_carbon_wall = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['jenkins_CWall_clear.txt', 'jenkins_CFC_unint.txt'],
        'Subset of jet carbon wall data for Jenkins tests')
    jenkins_jet_iterlike_wall = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['jenkins_ILW_unint.txt', 'jenkins_BeWall_clear.txt'],
        'Subset of jet iter like wall data for Jenkins tests')

    jet_full = ShotListFiles(
        sig.jet, params['paths']['shot_list_dir'],
        ['ILW_unint.txt', 'BeWall_clear.txt', 'CWall_clear.txt',
            'CFC_unint.txt'], 'jet full data')

    # d3d_10000 = ShotListFiles(
    #     sig.d3d, params['paths']['shot_list_dir'],
    #     ['d3d_clear_10000.txt', 'd3d_disrupt_10000.txt'],
    #     'd3d data 10000 ND and D shots')
    # d3d_1000 = ShotListFiles(
    #     sig.d3d, params['paths']['shot_list_dir'],
    #     ['d3d_clear_1000.txt', 'd3d_disrupt_1000.txt'],
    #     'd3d data 1000 ND and D shots')
    # d3d_100 = ShotListFiles(
    #     sig.d3d, params['paths']['shot_list_dir'],
    #     ['d3d_clear_100.txt', 'd3d_disrupt_100.txt'],
    #     'd3d data 100 ND and D shots')
    d3d_full = ShotListFiles(
        sig.d3d, params['paths']['shot_list_dir'],
        ['d3d_clear_data_avail.txt', 'd3d_disrupt_data_avail.txt'],
        'd3d data since shot 125500')  # to 168555
    # superset of d3d_full added in 2019 from C. Rea:
    d3d_full_2019 = ShotListFiles(
        sig.d3d, params['paths']['shot_list_dir'],
        ['d3d_clear_since_2016.txt', 'd3d_disrupt_since_2016.txt'],
        'd3d data since shot 125500')  # to 180847
    d3d_jenkins = ShotListFiles(
        sig.d3d, params['paths']['shot_list_dir'],
        ['jenkins_d3d_clear.txt', 'jenkins_d3d_disrupt.txt'],
        'Subset of d3d data for Jenkins test')

    # TODO(KGF): currently unused shot list files in project directory
    # /tigress/FRNN/shot_lists/:
    # d3d_clear.txt : 40560, 168554
    # d3d_disrupt   : 100000, 168555

    # TODO(KGF): should /tigress/FRNN/shot_lists/ be organized into subdirs
    # like the original repo directory data/shot_lists/d3d/, jet/, nstx/ ?

    # d3d_jb_full = ShotListFiles(
    #     sig.d3d, params['paths']['shot_list_dir'],
    #     ['shotlist_JaysonBarr_clear.txt',
    #      'shotlist_JaysonBarr_disrupt.txt'],
    #     'd3d shots since 160000-170000')

    # nstx_full = ShotListFiles(
    #     nstx, params['paths']['shot_list_dir'],
    #     ['disrupt_nstx.txt'], 'nstx shots (all are disruptive')
    # ==================
    # JET DATASETS
    # ==================
    if params['paths']['data'] == 'jet_all':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.jet_signals
    elif params['paths']['data'] == 'jet_0D':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.jet_signals_0D
    elif params['paths']['data'] == 'jet_1D':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.jet_signals_1D
    elif params['paths']['data'] == 'jet_late':
        params['paths']['shot_files'] = [jet_iterlike_wall_late]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = sig.jet_signals
    elif params['paths']['data'] == 'jet_carbon_to_late_0D':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall_late]
        params['paths']['use_signals_dict'] = sig.jet_signals_0D
    elif params['paths']['data'] == 'jet_temp_profile':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = {
            'etemp_profile': sig.etemp_profile}
    elif params['paths']['data'] == 'jet_dens_profile':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = {
            'edens_profile': sig.edens_profile}
    elif params['paths']['data'] == 'jet_carbon_all':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = sig.jet_signals
    elif params['paths']['data'] == 'jet_mixed_all':
        params['paths']['shot_files'] = [jet_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = sig.jet_signals
    elif params['paths']['data'] == 'jenkins_jet':
        params['paths']['shot_files'] = [jenkins_jet_carbon_wall]
        params['paths']['shot_files_test'] = [jenkins_jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.jet_signals
    # JET data but with fully defined signals
    elif params['paths']['data'] == 'jet_fully_defined':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals
    # JET data but with fully defined signals
    elif params['paths']['data'] == 'jet_fully_defined_0D':
        params['paths']['shot_files'] = [jet_carbon_wall]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
    # ==================
    # D3D DATASETS
    # ==================
    elif params['paths']['data'] == 'd3d_all':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95': sig.q95,
            'li': sig.li,
            'ip': sig.ip,
            'lm': sig.lm,
            'betan': sig.betan,
            'energy': sig.energy,
            'dens': sig.dens,
            'pradcore': sig.pradcore,
            'pradedge': sig.pradedge,
            'pin': sig.pin,
            'torquein': sig.torquein,
            'ipdirect': sig.ipdirect,
            'iptarget': sig.iptarget,
            'iperr': sig.iperr,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
        }
    elif params['paths']['data'] in ['d3d_data_max_tol',
                                        'd3d_data_garbage']:
        params['paths']['shot_files'] = [d3d_full_2019]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95t': sig.q95t,
            'lit': sig.lit,
            'ipt': sig.ipt,
            'lmt': sig.lmt,
            'betant': sig.betant,
            'energyt': sig.energyt,
            'denst': sig.denst,
            'pradcoret': sig.pradcoret,
            'pradedget': sig.pradedget,
            'pint': sig.pint,
            'torqueint': sig.torqueint,
            'ipdirectt': sig.ipdirectt,
            'iptargett': sig.iptargett,
            'iperrt': sig.iperrt,
            'etemp_profilet': sig.etemp_profilet,
            'edens_profilet': sig.edens_profilet,
        }
    elif params['paths']['data'] == 'd3d_2019':
        params['paths']['shot_files'] = [d3d_full_2019]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95': sig.q95,
            'li': sig.li,
            'ip': sig.ip,
            'lm': sig.lm,
            'betan': sig.betan,
            'energy': sig.energy,
            'dens': sig.dens,
            'pradcore': sig.pradcore,
            'pradedge': sig.pradedge,
            'pin': sig.pin,
            'torquein': sig.torquein,
            'ipdirect': sig.ipdirect,
            'iptarget': sig.iptarget,
            'iperr': sig.iperr,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
        }
    elif params['paths']['data'] == 'd3d_2019_all_dims':
        params['paths']['shot_files'] = [d3d_full_2019]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95': sig.q95,
            'li': sig.li,
            'ip': sig.ip,
            'lm': sig.lm,
            'betan': sig.betan,
            'energy': sig.energy,
            'dens': sig.dens,
            'pradcore': sig.pradcore,
            'pradedge': sig.pradedge,
            'pin': sig.pin,
            'torquein': sig.torquein,
            'ipdirect': sig.ipdirect,
            'iptarget': sig.iptarget,
            'iperr': sig.iperr,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
            'ecei': sig.ecei,
        }
    elif params['paths']['data'] == 'd3d_1D':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'ipdirect': sig.ipdirect,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
        }
    elif params['paths']['data'] == 'd3d_2D':
        params['paths']['shot_files'] = [d3d_full_2019]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'ecei': sig.ecei,
        }
    elif params['paths']['data'] == 'd3d_all_profiles':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'ipdirect': sig.ipdirect,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
            'itemp_profile': sig.itemp_profile,
            'zdens_profile': sig.zdens_profile,
            'trot_profile': sig.trot_profile,
            'pthm_profile': sig.pthm_profile,
            'neut_profile': sig.neut_profile,
            'q_profile': sig.q_profile,
            'bootstrap_current_profile': sig.bootstrap_current_profile,
            'q_psi_profile': sig.q_psi_profile,
        }
    elif params['paths']['data'] == 'd3d_0D':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95': sig.q95,
            'li': sig.li,
            'ip': sig.ip,
            'lm': sig.lm,
            'betan': sig.betan,
            'energy': sig.energy,
            'dens': sig.dens,
            'pradcore': sig.pradcore,
            'pradedge': sig.pradedge,
            'pin': sig.pin,
            'torquein': sig.torquein,
            'ipdirect': sig.ipdirect,
            'iptarget': sig.iptarget,
            'iperr': sig.iperr,
        }
    # TODO(KGF): rename. Unlike JET, there are probably differences between
    # sig.d3d_signals and the manually-defined sigs in above d3d_all
    # elif params['paths']['data'] == 'd3d_all':
    #     params['paths']['shot_files'] = [d3d_full]
    #     params['paths']['shot_files_test'] = []
    #     params['paths']['use_signals_dict'] = sig.d3d_signals
    elif params['paths']['data'] == 'jenkins_d3d':
        params['paths']['shot_files'] = [d3d_jenkins]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'q95': sig.q95,
            'li': sig.li,
            'ip': sig.ip,
            'lm': sig.lm,
            'betan': sig.betan,
            'energy': sig.energy,
            'dens': sig.dens,
            'pradcore': sig.pradcore,
            'pradedge': sig.pradedge,
            'pin': sig.pin,
            'torquein': sig.torquein,
            'ipdirect': sig.ipdirect,
            'iptarget': sig.iptarget,
            'iperr': sig.iperr,
            'etemp_profile': sig.etemp_profile,
            'edens_profile': sig.edens_profile,
        }
    # jet data but with fully defined signals
    elif params['paths']['data'] == 'd3d_fully_defined':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = sig.fully_defined_signals
    # jet data but with fully defined signals
    elif params['paths']['data'] == 'd3d_fully_defined_0D':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
    elif params['paths']['data'] == 'd3d_temp_profile':
        # jet data but with fully defined signals
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'etemp_profile': sig.etemp_profile}  # fully_defined_signals_0D
    elif params['paths']['data'] == 'd3d_dens_profile':
        # jet data but with fully defined signals
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = []
        params['paths']['use_signals_dict'] = {
            'edens_profile': sig.edens_profile}  # fully_defined_signals_0D
    # ======================
    # CROSS-MACHINE DATASETS
    # ======================
    elif params['paths']['data'] == 'jet_to_d3d_all':
        params['paths']['shot_files'] = [jet_full]
        params['paths']['shot_files_test'] = [d3d_full]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals
    elif params['paths']['data'] == 'd3d_to_jet_all':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals
    elif params['paths']['data'] == 'd3d_to_late_jet':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = [jet_iterlike_wall_late]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals
    elif params['paths']['data'] == 'jet_to_d3d_0D':
        params['paths']['shot_files'] = [jet_full]
        params['paths']['shot_files_test'] = [d3d_full]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
    elif params['paths']['data'] == 'd3d_to_jet_0D':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_0D
    elif params['paths']['data'] == 'jet_to_d3d_1D':
        params['paths']['shot_files'] = [jet_full]
        params['paths']['shot_files_test'] = [d3d_full]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_1D
    elif params['paths']['data'] == 'd3d_to_jet_1D':
        params['paths']['shot_files'] = [d3d_full]
        params['paths']['shot_files_test'] = [jet_iterlike_wall]
        params['paths']['use_signals_dict'] = sig.fully_defined_signals_1D

    else:
        g.print_unique("Unknown dataset {}".format(
            params['paths']['data']))
        exit(1)

    if len(params['paths']['specific_signals']):
        for s in params['paths']['specific_signals']:
            if s not in params['paths']['use_signals_dict'].keys():
                g.print_unique(
                    "Signal {} is not fully defined for {} machine. ",
                    "Skipping...".format(
                        s, params['paths']['data'].split("_")[0]))
        params['paths']['specific_signals'] = list(
            filter(
                lambda x: x in params['paths']['use_signals_dict'].keys(),
                params['paths']['specific_signals']))
        selected_signals = {k: params['paths']['use_signals_dict'][k]
                            for k in params['paths']['specific_signals']}
        params['paths']['use_signals'] = sort_by_channels(
            list(selected_signals.values()))
    else:
        # default case
        params['paths']['use_signals'] = sort_by_channels(
            list(params['paths']['use_signals_dict'].values()))

    params['paths']['all_signals'] = sort_by_channels(
        list(params['paths']['all_signals_dict'].values()))

    g.print_unique("Selected signals (determines which signals are used"
                    + " for training):\n{}".format(
                        params['paths']['use_signals']))
    params['paths']['shot_files_all'] = (
        params['paths']['shot_files'] + params['paths']['shot_files_test'])
    params['paths']['all_machines'] = list(
        set([file.machine for file in params['paths']['shot_files_all']]))

    # type assertations
    assert (isinstance(params['data']['signal_to_augment'], str)
            or isinstance(params['data']['signal_to_augment'], None))
    assert isinstance(params['data']['augment_during_training'], bool)

    return params


def sort_by_channels(list_of_signals):
    # make sure 1D signals come last! This is necessary for model builder.
    return sorted(list_of_signals, key=lambda x: x.num_channels)