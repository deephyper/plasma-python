import os
import logging
import traceback

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
gpu_per_node = 8
gpu_local_idx = rank % gpu_per_node
node = int(rank / gpu_per_node)

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logging.info(f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        logging.info(f"{e}")

import pathlib
import numpy as np
import gc
import time
import pprint
from functools import partial
import json

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import (
    Input,
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    LSTM, SimpleRNN, BatchNormalization,
    Convolution1D, MaxPooling1D, TimeDistributed,
    Concatenate
    )
CuDNNLSTM = LSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import (SGD, Adam, RMSprop, Nadam)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2  # l1, l1_l2
from tensorflow.keras.metrics import AUC

# from plasma.models.tcn import TCN
from plasma.conf_parser import parameters
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader
from plasma.utils.performance import PerformanceAnalyzer


class CustomModel(tf.keras.Model):
    
    def state_reset(self, batches_to_reset):
        for layer in (layer for layer in self.layers if hasattr(layer, 'states') and getattr(layer, 'stateful', False)):
            for i, state in enumerate(layer.states):
                assert state.shape[0] == batches_to_reset.shape[0]
                stacked_batches_to_reset = tf.stack([batches_to_reset for _ in range(state.shape[1])], axis=1)
                new_val = tf.multiply(state, 1-stacked_batches_to_reset)
                layer.states[i].assign(new_val)
        return batches_to_reset
        
    def train_step(self, data):
        x, r, y = data

        tf.cond(
            tf.greater(tf.reduce_sum(r), 0),
            lambda: self.state_reset(r),
            lambda: r
        )
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
    

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, r, y = data

        tf.cond(
            tf.greater(tf.reduce_sum(r), 0),
            lambda: self.state_reset(r),
            lambda: r
        )

        y_pred = self(x, training=False)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
    
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class ModelBuilder(object):
    def __init__(self, conf):
        self.conf = conf

    def get_0D_1D_indices(self):
        # make sure all 1D indices are contiguous in the end!
        use_signals = self.conf['paths']['use_signals']
        indices_0d = []
        indices_1d = []
        num_0D = 0
        num_1D = 0
        curr_idx = 0
        is_1D_region = use_signals[0].num_channels > 1
        for sig in use_signals:
            num_channels = sig.num_channels
            indices = range(curr_idx, curr_idx+num_channels)
            if num_channels > 1:
                indices_1d += indices
                num_1D += 1
                is_1D_region = True
            else:
                assert not is_1D_region, ("Check that use_signals are ordered with 1D signals last!")
                assert num_channels == 1
                indices_0d += indices
                num_0D += 1
                is_1D_region = False
            curr_idx += num_channels
        return (np.array(indices_0d).astype(np.int32), np.array(indices_1d).astype(np.int32), num_0D, num_1D)

    def build_model(self):
        conf = self.conf
        model_conf = conf['model']
        rnn_size = model_conf['rnn_size']
        rnn_type = model_conf['rnn_type']
        regularization = model_conf['regularization']
        dense_regularization = model_conf['dense_regularization']
        use_batch_norm = model_conf.get('use_batch_norm', False)

        dropout_prob = model_conf['dropout_prob']
        length = model_conf['length']
        stateful = model_conf['stateful']
        return_sequences = model_conf['return_sequences']
        output_activation = targets[conf['target']]['activation']
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        num_conv_filters = model_conf['num_conv_filters']
        num_conv_layers = model_conf['num_conv_layers']
        size_conv_filters = model_conf['size_conv_filters']
        pool_size = model_conf['pool_size']
        dense_size = model_conf['dense_size']

        batch_size = conf['training']['batch_size']

        if rnn_type == 'LSTM':
            rnn_model = LSTM
        elif rnn_type == 'SimpleRNN':
            rnn_model = SimpleRNN
        else:
            rnn_model = CuDNNLSTM

        batch_input_shape = (batch_size, length, num_signals)

        indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

        # ==========
        # PRE_RNN
        # ==========
        pre_rnn_input = Input(shape=(num_signals,))

        if num_1D > 0:
            pre_rnn_1D = Lambda(lambda x: x[:, len(indices_0d):], output_shape=(len(indices_1d),))(pre_rnn_input)
            pre_rnn_0D = Lambda(lambda x: x[:, :len(indices_0d)], output_shape=(len(indices_0d),))(pre_rnn_input)
            pre_rnn_1D = Reshape((num_1D, len(indices_1d)//num_1D))(pre_rnn_1D)
            pre_rnn_1D = Permute((2, 1))(pre_rnn_1D)
            if model_conf.get('simple_conv', False):
                for i in range(num_conv_layers):
                    pre_rnn_1D = Convolution1D(num_conv_filters, size_conv_filters, padding='valid', activation='relu')(pre_rnn_1D)
                pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            else:
                for i in range(num_conv_layers):
                    div_fac = 2**i
                    pre_rnn_1D = Convolution1D(num_conv_filters//div_fac, size_conv_filters, padding='valid')(pre_rnn_1D)
                    if use_batch_norm:
                        pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                    pre_rnn_1D = Convolution1D(num_conv_filters//div_fac, 1, padding='valid')(pre_rnn_1D)
                    if use_batch_norm:
                        pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                    pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            pre_rnn_1D = Flatten()(pre_rnn_1D)
            pre_rnn_1D = Dense(dense_size, kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn_1D = Dense(dense_size//4, kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn = Concatenate()([pre_rnn_0D, pre_rnn_1D])
        else:
            pre_rnn = pre_rnn_input

        if model_conf['rnn_layers'] == 0 or model_conf.get('extra_dense_input', False):
            pre_rnn = Dense(dense_size, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(dense_size//2, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(dense_size//4, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)

        pre_rnn_model = tf.keras.Model(inputs=pre_rnn_input, outputs=pre_rnn)
        # pre_rnn_model.summary()
        

        x_input = Input(batch_shape=batch_input_shape)
        if num_1D > 0 or model_conf.get('extra_dense_input', False):
            x_in = TimeDistributed(pre_rnn_model)(x_input)
        else:
            x_in = x_input
        
        # ==========
        # RNN MODEL
        # ==========
        model_kwargs = dict(return_sequences=return_sequences,
                            # batch_input_shape=batch_input_shape,
                            stateful=stateful,
                            kernel_regularizer=l2(regularization),
                            recurrent_regularizer=l2(regularization),
                            bias_regularizer=l2(regularization),
                            )
        if rnn_type != 'CuDNNLSTM':
            # recurrent_dropout is unsupported in cuDNN library
            model_kwargs['recurrent_dropout'] = dropout_prob  # recurrent states
        model_kwargs['dropout'] = dropout_prob  # input states
        for _ in range(model_conf['rnn_layers']):
            rnn_layer = rnn_model(rnn_size, **model_kwargs)
            x_in = rnn_layer(x_in)
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            x_out = TimeDistributed(Dense(1, activation=output_activation))(x_in)

        model = CustomModel(inputs=x_input, outputs=x_out)
        # bug with tensorflow/Keras
        # TODO(KGF): what is this bug? this is the only direct "tensorflow"
        # import outside of mpi_runner.py and runner.py
        # if (conf['model']['backend'] == 'tf'
        #         or conf['model']['backend'] == 'tensorflow'):
        #     first_time = "tensorflow" not in sys.modules
        #     import tensorflow as tf
        #     if first_time:
        #         tf.compat.v1.keras.backend.get_session().run(
        #             tf.global_variables_initializer())
        model.reset_states()
        return model

    def build_optimizer(self):
        conf = self.conf
        lr = conf['model']["lr"]
        momentum = conf['model']["momentum"]
        clipnorm = conf['model']["clipnorm"]
        optimizers = {
            "adam": Adam,
        }
        optimizers_kwargs = {
            "adam": {
                "learning_rate": lr,
                "epsilon": 1e-8,
            },
        }
        
        opt_kwargs = optimizers_kwargs.get(conf['model']['optimizer'], optimizers_kwargs['adam'])
        optimizer = optimizers.get(conf['model']['optimizer'], optimizers['adam'])(**opt_kwargs)

        return optimizer


class DataHandler(object):
    def __init__(self, conf):
        self.conf = conf
    
    def build_loader(self):
        conf = self.conf
        if conf['data']['normalizer'] == 'minmax':
            from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
        elif conf['data']['normalizer'] == 'meanvar':
            from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
        elif conf['data']['normalizer'] == 'averagevar':
            from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer
        else: # conf['data']['normalizer'] == 'var':
            from plasma.preprocessor.normalize import VarNormalizer as Normalizer

        normalizer = Normalizer(conf)
        normalizer.train()
        loader = Loader(conf, normalizer)
        loader.verbose = False
        return loader

    def load_dataset(self, shot_list, loader):
        conf = self.conf
        batch_generator = partial(loader.training_batch_generator_partial_reset_bis, shot_list=shot_list)

        length = conf['model']['length']
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        batch_size = conf['training']['batch_size']
        
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, length, num_signals), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, length, 1), dtype=tf.float32),
            )
        )
        return dataset


class HingeLoss(tf.losses.Hinge):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, None)

class CrossLoss(tf.losses.BinaryCrossentropy):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, 0, 1)
        return super().__call__(y_true, y_pred, None)

class FocalLoss(tf.losses.BinaryFocalCrossentropy):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, 0, 1)
        return super().__call__(y_true, y_pred, None)

class BalancedHingeLoss(tf.losses.Hinge):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        sample_weight = (1-y_true)*self._class_weight[0] + y_true*self._class_weight[1]
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

class BalancedCrossLoss(tf.losses.BinaryCrossentropy):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        # frnn y_true are either -1 or 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        sample_weight = (1-y_true)*self._class_weight[0] + y_true*self._class_weight[1]
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

class BalancedFocalLoss(tf.losses.BinaryFocalCrossentropy):
    def set_class_weight(self, class_weight):
        self._class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        # frnn y_true are either -1 or 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        sample_weight = (1-y_true)*self._class_weight[0] + y_true*self._class_weight[1]
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class AdaptedAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, 0, 1)
        return super().update_state(y_true, y_pred, sample_weight)


class ModelEvaluator(object):
    def __init__(self, model, loader, conf) -> None:
        self._model = model
        self._loader = loader
        self._conf = conf
    
    def update_model(self, model):
        self._model = model

    def make_predictions(self, shot_list):
        self._loader.set_inference_mode(True)
        shot_list.sort()

        self._model.reset_states()
        shot_sublists = shot_list.sublists(self._conf['model']['pred_batch_size'], do_shuffle=False, equal_size=True)
        y_prime = []
        y_gold = []
        disruptive = []

        for shot_sublist in shot_sublists:

            X, y, shot_lengths, disr = self._loader.load_as_X_y_pred(shot_sublist)

            # load data and fit on data
            self._model.reset_states()
            y_p = self._model.predict(X, batch_size=self._conf['model']['pred_batch_size'])

            y_p = self._loader.batch_output_to_array(y_p)
            y = self._loader.batch_output_to_array(y)

            # cut arrays back
            y_p = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y_p)]
            y = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y)]

            y_prime += y_p
            y_gold += y
            disruptive += disr

        y_prime = y_prime[:len(shot_list)]
        y_gold = y_gold[:len(shot_list)]
        disruptive = disruptive[:len(shot_list)]
        self._loader.set_inference_mode(False)
        return y_prime, y_gold, disruptive

    def make_predictions_and_evaluate(self, shot_list):
        y_prime, y_gold, disruptive = self.make_predictions(shot_list)
        analyzer = PerformanceAnalyzer(conf=self._conf)
        roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
        # shot_list.set_weights(analyzer.get_shot_difficulty(y_prime, y_gold, disruptive))
        # loss = get_loss_from_list(y_prime, y_gold, self._conf['data']['target'])
        return roc_area #, loss


class FrnnEvaluatorCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, loader, shot_list_train, shot_list_valid, name, path, conf) -> None:
        super().__init__()
        self._shot_list_train = shot_list_train
        self._shot_list_valid = shot_list_valid
        self._evaluator = ModelEvaluator(model, loader, conf)
        self._t_start = time.time()
        self._name = name
        self._path = path
        self._history = dict()
        self._best_valid_roc = 0

    def on_epoch_end(self, epoch, logs=None):
        self._evaluator.update_model(self.model)
        train_roc = self._evaluator.make_predictions_and_evaluate(self._shot_list_train)
        valid_roc = self._evaluator.make_predictions_and_evaluate(self._shot_list_valid)
        logs['frnn_roc'] = train_roc
        logs['val_frnn_roc'] = valid_roc
        logs['t_epoch_end'] = time.time() - self._t_start
        for metric, value in logs.items(): 
            if metric != 'lr':
                val_list = self._history.get(metric, [])
                val_list.append(value)
                self._history[metric] = val_list
        with open(f"{self._path}/histories/{self._name}.json", 'w') as file:
            json.dump(self._history, file)
        if valid_roc > self._best_valid_roc:
            self._best_valid_roc = valid_roc
            self.model.save_weights(f"{self._path}/model_weights/{self._name}.h5")

        gc.collect()
        tf.keras.backend.clear_session()


targets = {
    'hinge': {
        'loss': HingeLoss(),
        'activation': 'linear',
    },
    'cross': {
        'loss': CrossLoss(from_logits=True),
        'activation': 'linear',
    },
    'focal': {
        'loss': FocalLoss(from_logits=True),
        'activation': 'linear',
    },
    'balanced_hinge': {
        'loss': BalancedHingeLoss(),
        'activation': 'linear',
    },
    'balanced_cross': {
        'loss': BalancedCrossLoss(from_logits=True),
        'activation': 'linear',
    },
    'balanced_focal': {
        'loss': BalancedFocalLoss(from_logits=True),
        'activation': 'linear',
    },
}


def run(config: None, results_path, model_name):
    try:
        clear_session()

        # config['fs_path'] = f"/dev/shm/{node}"
        # pp = pprint.PrettyPrinter(indent=1)
        # pp.pprint(config)

        conf = parameters(config)

        handler = DataHandler(conf)
        builder = ModelBuilder(conf)

        # load data
        (shot_list_train, shot_list_valid, shot_list_test) = guarantee_preprocessed(conf)

        loader = handler.build_loader()
        train_dataset = handler.load_dataset(shot_list_train, loader)
        valid_dataset = handler.load_dataset(shot_list_valid, loader)
    
        # build model
        model = builder.build_model()
        # model.summary()

        signals, results, shot_lengths, disruptive = loader.get_signals_results_from_shotlist(shot_list_valid, prediction_mode=True)
        N = len(disruptive)
        n_p = np.sum(disruptive)
        n_n = N - n_p
        coeff_p = N/(2*n_p)
        coeff_n = N/(2*n_n)
        class_weight = {0: N/(coeff_n*2), 1: N/(coeff_p*2)}

        loss = targets[conf['target']]['loss']
        loss.set_class_weight(class_weight)

        optimizer = builder.build_optimizer()
        model.compile(optimizer=optimizer, loss=loss) #, metrics=[AdaptedAUC(name="auc", from_logits=True)])

        lr_decay = conf['model']['lr_decay']
        def scheduler(epoch, lr):
            if epoch > 0:
                lr *= lr_decay
            return lr

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        frnn_evaluator = FrnnEvaluatorCallback(model, loader, shot_list_train, shot_list_valid, model_name, results_path, conf)

        steps_per_epoch = loader.get_steps_per_epoch_bis(shot_list_train)
        validation_steps = loader.get_steps_per_epoch_bis(shot_list_valid)

        # train it
        num_epochs = 128
        history = model.fit(
            train_dataset,
            batch_size=conf['training']['batch_size'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            callbacks=[lr_scheduler, frnn_evaluator],
        )

        history_dict = history.history

        # evaluate it
        evaluator = ModelEvaluator(model, loader, conf)
        train_roc = history_dict['frnn_roc'][-1]
        valid_roc = history_dict['val_frnn_roc'][-1]
        test_roc = evaluator.make_predictions_and_evaluate(shot_list_test)

        # print results
        print(f'[{rank}]======== RESULTS =======')
        print(f'[{rank}]Train ROC: {train_roc:.4f}')
        print(f'[{rank}]Valid ROC: {valid_roc:.4f}')
        print(f'[{rank}]Test ROC: {test_roc:.4f}')

        objective = valid_roc

        clear_session()
    
    except tf.errors.ResourceExhaustedError as e:
        print("ERROR DURING RUN FUNCTION :")
        print(traceback.format_exc())
        objective = 0
        clear_session()

    return objective

if __name__ == '__main__':
    path_to_top_configs = "configs/top_80.json"
    results_path = "results/top_80"
    model_name = f"top_{rank+1}"
    with open(path_to_top_configs, 'r') as file:
        topk_conf = json.load(file)
    t = time.time()
    obj = run(topk_conf[rank], results_path, model_name)
    print(f"[{rank}] result: {obj}")
    print(f"[{rank}] Run duration : {time.time() - t:.2f}s.")