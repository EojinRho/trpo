"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import json


class NNValueFunctionContinue(object):
    """ NN-based state-value function """
    def __init__(self, filename, obs_dim, net_size_factor=10):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epochs = 10
        self.net_size_factor = net_size_factor
        self.lr = None  # learning rate set in _build_graph()

        with open(filename) as f:
            data = json.load(f)

        self.w0 = np.array(data[0][1][0])
        self.b0 = np.array(data[0][1][1])
        self.w1 = np.array(data[0][1][2])
        self.b1 = np.array(data[0][1][3])
        self.w2 = np.array(data[0][1][4])
        self.b2 = np.array(data[0][1][5])
        self.w3 = np.array(data[0][1][6])
        self.b3 = np.array(data[0][1][7])

        self._build_graph()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.net_size_factor  # 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            init_w = tf.constant_initializer(self.w0)
            init_b = tf.constant_initializer(self.b0)
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="h1")
            init_w = tf.constant_initializer(self.w1)
            init_b = tf.constant_initializer(self.b1)
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                               kernel_initializer=init_w, bias_initializer=init_b, name="h2")
            init_w = tf.constant_initializer(self.w2)
            init_b = tf.constant_initializer(self.b2)
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                               kernel_initializer=init_w, bias_initializer=init_b, name="h3")
            init_w = tf.constant_initializer(self.w3)
            init_b = tf.constant_initializer(self.b3)
            out = tf.layers.dense(out, 1,
                               kernel_initializer=init_w, bias_initializer=init_b, name="output")
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

        # store weights
        model_list_names, model_list_params = self.get_model_as_list()
        model_list = [[model_list_names, model_list_params]]
        logger.log_model_2(model_list)

    def get_model_as_list(self):
        # get trainable params.
        model_names = []
        model_params = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                params = self.sess.run(var).tolist()
                model_names.append(param_name)
                model_params.append(params)
        return model_names, model_params

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
