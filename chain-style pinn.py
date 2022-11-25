import tensorflow as tf
import numpy as np
import pandas as pd
import time
import sys
import os


def tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))


def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


def load_data(path):
    all_data_list = os.listdir(path)
    if len(all_data_list[0]) == 21:
        all_data_list.sort(key=lambda x: int(x[-12:-6]))
    if len(all_data_list[0]) == 22:
        all_data_list.sort(key=lambda x: int(x[-13:-7]))
    for single_data in all_data_list:
        data_dir = path + single_data
        dataname = os.path.splitext(single_data)[0]
        data = np.array(pd.read_csv(data_dir, header=None), dtype=np.float32)
        if single_data == all_data_list[0]:
            U = data
        else:
            U = np.hstack((U, data))
    return U


class pinn_1(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        with tf.variable_scope('pinn_1'):
            for l in range(0, self.num_layers - 1):
                in_dim = self.layers[l]
                out_dim = self.layers[l + 1]
                W = np.random.normal(size=[in_dim, out_dim])
                b = np.zeros([1, out_dim])
                g = np.ones([1, out_dim])

                self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True, name='1_w' + str(l)))
                self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True, name='1_b' + str(l)))
                self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True, name='1_g' + str(l)))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]

            V = W / tf.norm(W, axis=0, keepdims=True)
            H = tf.matmul(H, V)
            H = g * H + b

            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1, name='my_pinn_1')

        return Y


class pinn_2(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        with tf.variable_scope('pinn_2'):
            for l in range(0, self.num_layers - 1):
                in_dim = self.layers[l]
                out_dim = self.layers[l + 1]
                W = np.random.normal(size=[in_dim, out_dim])
                b = np.zeros([1, out_dim])
                g = np.ones([1, out_dim])

                self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True, name='2_w' + str(l)))
                self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True, name='2_b' + str(l)))
                self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True, name='2_g' + str(l)))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]

            V = W / tf.norm(W, axis=0, keepdims=True)
            H = tf.matmul(H, V)
            H = g * H + b

            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)
            if l == self.num_layers - 2:
                H1 = H[:, :-1]
                H2 = tf.expand_dims(H[:, -1], axis=1)
                H3 = tf.sin(H2)
                H = tf.concat([H1, H3], axis=1)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1, name='my_pinn_2')

        return Y


def cavitation_governing_equations_3D(u, v, w, p, f, t, x, y, z):
    Y = tf.concat([u, v, w, p, f], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)

    u = Y[:, 0]
    v = Y[:, 1]
    w = Y[:, 2]
    p = Y[:, 3]
    f = Y[:, 4]

    u_t = Y_t[:, 0]
    v_t = Y_t[:, 1]
    w_t = Y_t[:, 2]
    p_t = Y_t[:, 3]
    f_t = Y_t[:, 4]

    u_x = Y_x[:, 0]
    v_x = Y_x[:, 1]
    w_x = Y_x[:, 2]
    p_x = Y_x[:, 3]
    f_x = Y_x[:, 4]

    u_y = Y_y[:, 0]
    v_y = Y_y[:, 1]
    w_y = Y_y[:, 2]
    p_y = Y_y[:, 3]
    f_y = Y_y[:, 4]

    u_z = Y_z[:, 0]
    v_z = Y_z[:, 1]
    w_z = Y_z[:, 2]
    p_z = Y_z[:, 3]
    f_z = Y_z[:, 4]

    u_xx = Y_xx[:, 0]
    v_xx = Y_xx[:, 1]
    w_xx = Y_xx[:, 2]

    u_yy = Y_yy[:, 0]
    v_yy = Y_yy[:, 1]
    w_yy = Y_yy[:, 2]

    u_zz = Y_zz[:, 0]
    v_zz = Y_zz[:, 1]
    w_zz = Y_zz[:, 2]

    r_air = 0.01
    r_water = 0.997
    m_air = 0.000016
    m_water = 0.000001

    r = f * r_water + (1 - f) * r_air
    m = (f * m_water + (1 - f) * m_air) * 20.

    e1 = 0.1 * (p_t + u * p_x + v * p_y + w * p_z) + r * (u_x + v_y + w_z)
    e2 = r * (u_t + u * u_x + v * u_y + w * u_z) + p_x - r * m * (u_xx + u_yy + u_zz)
    e3 = r * (v_t + u * v_x + v * v_y + w * v_z) + p_y - r * m * (v_xx + v_yy + v_zz)
    e4 = r * (w_t + u * w_x + v * w_y + w * w_z) + p_z - r * m * (w_xx + w_yy + w_zz)
    e5 = f_t + u * f_x + v * f_y + w * f_z - (10 - 9 * f) * (p - 0.1)

    return e1, e2, e3, e4, e5


class chain_style_pinn(object):
    def __init__(self, t_obs, x_obs, y_obs, z_obs, u_obs, v_obs, w_obs,
                 t_bdy, x_bdy, y_bdy, z_bdy, u_bdy, v_bdy, w_bdy, p_bdy, f_bdy,
                 t_eqns, x_eqns, y_eqns, z_eqns,
                 layers, batch_size):

        self.layers = layers
        self.batch_size = batch_size

        [self.t_obs,
         self.x_obs,
         self.y_obs,
         self.z_obs,
         self.u_obs,
         self.v_obs,
         self.w_obs] = [t_obs,
                        x_obs,
                        y_obs,
                        z_obs,
                        u_obs,
                        v_obs,
                        w_obs]

        [self.t_eqns,
         self.x_eqns,
         self.y_eqns,
         self.z_eqns] = [t_eqns,
                         x_eqns,
                         y_eqns,
                         z_eqns]
        [self.t_bdy,
         self.x_bdy,
         self.y_bdy,
         self.z_bdy,
         self.u_bdy,
         self.v_bdy,
         self.w_bdy,
         self.p_bdy,
         self.f_bdy] = [t_bdy,
                        x_bdy,
                        y_bdy,
                        z_bdy,
                        u_bdy,
                        v_bdy,
                        w_bdy,
                        p_bdy,
                        f_bdy]

        self.t_obs_tf = tf.placeholder(tf.float32, shape=[None, 1], name='t')
        self.x_obs_tf = tf.placeholder(tf.float32, shape=[None, 1], name='x')
        self.y_obs_tf = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.z_obs_tf = tf.placeholder(tf.float32, shape=[None, 1], name='z')
        self.u_obs_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_obs_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.w_obs_tf = tf.placeholder(tf.float32, shape=[None, 1])

        [self.t_eqns_tf,
         self.x_eqns_tf,
         self.y_eqns_tf,
         self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]

        [self.t_bdy_tf,
         self.x_bdy_tf,
         self.y_bdy_tf,
         self.z_bdy_tf,
         self.u_bdy_tf,
         self.v_bdy_tf,
         self.w_bdy_tf,
         self.p_bdy_tf,
         self.f_bdy_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(9)]

        self.pinn_1 = pinn_1(self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns, layers=self.layers)
        self.pinn_2 = pinn_2(self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns, layers=self.layers)

        [self.u_obs_pred_1,
         self.v_obs_pred_1,
         self.w_obs_pred_1,
         self.p_obs_pred_1,
         self.f_obs_pred_1] = self.pinn_1(self.t_obs_tf,
                                          self.x_obs_tf,
                                          self.y_obs_tf,
                                          self.z_obs_tf)

        [self.u_obs_pred_2,
         self.v_obs_pred_2,
         self.w_obs_pred_2,
         self.p_obs_pred_2,
         self.f_obs_pred_2] = self.pinn_2(self.t_obs_tf,
                                          self.x_obs_tf,
                                          self.y_obs_tf,
                                          self.z_obs_tf)

        [self.u_bdy_pred_1,
         self.v_bdy_pred_1,
         self.w_bdy_pred_1,
         self.p_bdy_pred_1,
         self.f_bdy_pred_1] = self.pinn_1(self.t_bdy_tf,
                                          self.x_bdy_tf,
                                          self.y_bdy_tf,
                                          self.z_bdy_tf)

        [self.u_bdy_pred_2,
         self.v_bdy_pred_2,
         self.w_bdy_pred_2,
         self.p_bdy_pred_2,
         self.f_bdy_pred_2] = self.pinn_2(self.t_bdy_tf,
                                          self.x_bdy_tf,
                                          self.y_bdy_tf,
                                          self.z_bdy_tf)

        [self.u_eqns_pred_1,
         self.v_eqns_pred_1,
         self.w_eqns_pred_1,
         self.p_eqns_pred_1,
         self.f_eqns_pred_1] = self.pinn_1(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf,
                                           self.z_eqns_tf)

        [self.u_eqns_pred_2,
         self.v_eqns_pred_2,
         self.w_eqns_pred_2,
         self.p_eqns_pred_2,
         self.f_eqns_pred_2] = self.pinn_2(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf,
                                           self.z_eqns_tf)

        [self.e1_eqns_pred_1,
         self.e2_eqns_pred_1,
         self.e3_eqns_pred_1,
         self.e4_eqns_pred_1,
         self.e5_eqns_pred_1] = cavitation_governing_equations_3D(self.u_eqns_pred_1,
                                                                  self.v_eqns_pred_1,
                                                                  self.w_eqns_pred_1,
                                                                  self.p_eqns_pred_1,
                                                                  self.f_eqns_pred_1,
                                                                  self.t_eqns_tf,
                                                                  self.x_eqns_tf,
                                                                  self.y_eqns_tf,
                                                                  self.z_eqns_tf)

        [self.e1_eqns_pred_2,
         self.e2_eqns_pred_2,
         self.e3_eqns_pred_2,
         self.e4_eqns_pred_2,
         self.e5_eqns_pred_2] = cavitation_governing_equations_3D(self.u_eqns_pred_2,
                                                                  self.v_eqns_pred_2,
                                                                  self.w_eqns_pred_2,
                                                                  self.p_eqns_pred_2,
                                                                  self.f_eqns_pred_2,
                                                                  self.t_eqns_tf,
                                                                  self.x_eqns_tf,
                                                                  self.y_eqns_tf,
                                                                  self.z_eqns_tf)

        # loss
        self.loss_obs_1 = mean_squared_error(self.u_obs_pred_1, self.u_obs_tf) + \
                          mean_squared_error(self.v_obs_pred_1, self.v_obs_tf) + \
                          mean_squared_error(self.w_obs_pred_1, self.w_obs_tf)

        self.loss_obs_2 = mean_squared_error(self.u_obs_pred_2, self.u_obs_tf) + \
                          mean_squared_error(self.v_obs_pred_2, self.v_obs_tf) + \
                          mean_squared_error(self.w_obs_pred_2, self.w_obs_tf) + \
                          mean_squared_error(self.p_obs_pred_2, self.p_obs_pred_1)

        self.loss_bdy_1 = mean_squared_error(self.u_bdy_pred_1, self.u_bdy_tf) + \
                          mean_squared_error(self.v_bdy_pred_1, self.v_bdy_tf) + \
                          mean_squared_error(self.w_bdy_pred_1, self.w_bdy_tf) + \
                          mean_squared_error(self.p_bdy_pred_1, self.p_bdy_tf) + \
                          mean_squared_error(self.f_bdy_pred_1, self.f_bdy_tf)

        self.loss_bdy_2 = mean_squared_error(self.u_bdy_pred_2, self.u_bdy_tf) + \
                          mean_squared_error(self.v_bdy_pred_2, self.v_bdy_tf) + \
                          mean_squared_error(self.w_bdy_pred_2, self.w_bdy_tf) + \
                          mean_squared_error(self.p_bdy_pred_2, self.p_bdy_tf) + \
                          mean_squared_error(self.f_bdy_pred_2, self.f_bdy_tf)

        self.loss_fun_1 = mean_squared_error(self.e1_eqns_pred_1, 0.0) + \
                          mean_squared_error(self.e2_eqns_pred_1, 0.0) + \
                          mean_squared_error(self.e3_eqns_pred_1, 0.0) + \
                          mean_squared_error(self.e4_eqns_pred_1, 0.0)

        self.loss_fun_2 = mean_squared_error(self.e1_eqns_pred_2, 0.0) + \
                          mean_squared_error(self.e2_eqns_pred_2, 0.0) + \
                          mean_squared_error(self.e3_eqns_pred_2, 0.0) + \
                          mean_squared_error(self.e4_eqns_pred_2, 0.0) + \
                          mean_squared_error(self.e5_eqns_pred_2, 0.0)

        self.loss_1 = self.loss_obs_1 + self.loss_bdy_1 + self.loss_fun_1
        self.loss_2 = self.loss_obs_2 + self.loss_bdy_2 + self.loss_fun_2

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        pinn_1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pinn_1')
        pinn_2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pinn_2')
        self.train_op_1 = self.optimizer.minimize(self.loss_1, var_list=pinn_1_params)
        self.train_op_2 = self.optimizer.minimize(self.loss_2, var_list=pinn_2_params)

        self.sess = tf_session()

    def train(self, total_iteration, learning_rate):
        N_obs = self.t_obs.shape[0]
        N_bdy = self.t_bdy.shape[0]
        N_eqns = self.t_eqns.shape[0]

        start_time = time.time()
        running_time = 0
        it = 0

        while True:
            if it > 5e4:
                learning_rate = 5e-4
            if it > 1e5:
                learning_rate = 1e-4
            if it > 1.5e5:
                learning_rate = 5e-5
            idx_obs = np.random.choice(N_obs, min(self.batch_size, N_obs))
            idx_bdy = np.random.choice(N_bdy, min(self.batch_size, N_bdy))
            idx_eqns = np.random.choice(N_eqns, min(self.batch_size, N_eqns))

            (t_obs_batch,
             x_obs_batch,
             y_obs_batch,
             z_obs_batch,
             u_obs_batch,
             v_obs_batch,
             w_obs_batch) = (self.t_obs[idx_obs, :],
                             self.x_obs[idx_obs, :],
                             self.y_obs[idx_obs, :],
                             self.z_obs[idx_obs, :],
                             self.u_obs[idx_obs, :],
                             self.v_obs[idx_obs, :],
                             self.w_obs[idx_obs, :])

            (t_bdy_batch,
             x_bdy_batch,
             y_bdy_batch,
             z_bdy_batch,
             u_bdy_batch,
             v_bdy_batch,
             w_bdy_batch,
             p_bdy_batch,
             f_bdy_batch) = (self.t_bdy[idx_bdy, :],
                             self.x_bdy[idx_bdy, :],
                             self.y_bdy[idx_bdy, :],
                             self.z_bdy[idx_bdy, :],
                             self.u_bdy[idx_bdy, :],
                             self.v_bdy[idx_bdy, :],
                             self.w_bdy[idx_bdy, :],
                             self.p_bdy[idx_bdy, :],
                             self.f_bdy[idx_bdy, :])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.t_eqns[idx_eqns, :],
                              self.x_eqns[idx_eqns, :],
                              self.y_eqns[idx_eqns, :],
                              self.z_eqns[idx_eqns, :])

            tf_dict = {self.t_obs_tf: t_obs_batch,
                       self.x_obs_tf: x_obs_batch,
                       self.y_obs_tf: y_obs_batch,
                       self.z_obs_tf: z_obs_batch,
                       self.u_obs_tf: u_obs_batch,
                       self.v_obs_tf: v_obs_batch,
                       self.w_obs_tf: w_obs_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.t_bdy_tf: t_bdy_batch,
                       self.x_bdy_tf: x_bdy_batch,
                       self.y_bdy_tf: y_bdy_batch,
                       self.z_bdy_tf: z_bdy_batch,
                       self.u_bdy_tf: u_bdy_batch,
                       self.v_bdy_tf: v_bdy_batch,
                       self.w_bdy_tf: w_bdy_batch,
                       self.p_bdy_tf: p_bdy_batch,
                       self.f_bdy_tf: f_bdy_batch,
                       self.learning_rate: learning_rate}

            # Save
            self.sess.run([self.train_op_1], tf_dict)
            if it % 1000 == 0:
                saver = tf.train.Saver(max_to_keep=1)
                saver.save(self.sess, 'chain-style pinn/pinn_1/pinn_1.ckpt')

            self.sess.run([self.train_op_2], tf_dict)
            if it % 1000 == 0:
                saver = tf.train.Saver(max_to_keep=1)
                saver.save(self.sess, 'chain-style pinn/pinn_2/pinn_2.ckpt')

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                [loss_value_1,
                 loss_value_2,
                 learning_rate_value] = self.sess.run([self.loss_1,
                                                       self.loss_2,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss 1: %.3e, Loss 2: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      % (it, loss_value_1, loss_value_2, elapsed, running_time, learning_rate_value))

                sys.stdout.flush()
                start_time = time.time()

            it += 1
            if it > 2e5:
                break


if __name__ == "__main__":
    batch_size = 10000

    layers = [4] + 10 * [500] + [5]

    # load data
    file_dir = 'cavitation data 3D/'
    t = np.array(pd.read_csv(file_dir + 'time.csv', header=None), dtype=np.float32)
    T = t.shape[0]

    # load observation data
    x_obs = np.array(pd.read_csv(file_dir + 'x_obs.csv', header=None), dtype=np.float32)
    y_obs = np.array(pd.read_csv(file_dir + 'y_obs.csv', header=None), dtype=np.float32)
    z_obs = np.array(pd.read_csv(file_dir + 'z_obs.csv', header=None), dtype=np.float32)
    u_obs = load_data(file_dir + 'observation/u_obs/')
    v_obs = load_data(file_dir + 'observation/v_obs/')
    w_obs = load_data(file_dir + 'observation/w_obs/')

    N_obs = x_obs.shape[0]

    t_obs = np.tile(t, (1, N_obs)).T
    x_obs = np.tile(x_obs, (1, T))
    y_obs = np.tile(y_obs, (1, T))
    z_obs = np.tile(z_obs, (1, T))

    t_obs = t_obs.flatten()[:, None]
    x_obs = x_obs.flatten()[:, None]
    y_obs = y_obs.flatten()[:, None]
    z_obs = z_obs.flatten()[:, None]
    u_obs = u_obs.flatten()[:, None]
    v_obs = v_obs.flatten()[:, None]
    w_obs = w_obs.flatten()[:, None]

    # load boundary data
    x_bdy = np.array(pd.read_csv(file_dir + 'x_bdy.csv', header=None), dtype=np.float32)
    y_bdy = np.array(pd.read_csv(file_dir + 'y_bdy.csv', header=None), dtype=np.float32)
    z_bdy = np.array(pd.read_csv(file_dir + 'z_bdy.csv', header=None), dtype=np.float32)
    u_bdy = load_data(file_dir + 'boundary/u_bdy/')
    v_bdy = load_data(file_dir + 'boundary/v_bdy/')
    w_bdy = load_data(file_dir + 'boundary/w_bdy/')
    p_bdy = load_data(file_dir + 'boundary/p_bdy/')
    f_bdy = load_data(file_dir + 'boundary/f_bdy/')

    N_bdy = x_bdy.shape[0]

    t_bdy = np.tile(t, (1, N_bdy)).T
    x_bdy = np.tile(x_bdy, (1, T))
    y_bdy = np.tile(y_bdy, (1, T))
    z_bdy = np.tile(z_bdy, (1, T))

    t_bdy = t_bdy.flatten()[:, None]
    x_bdy = x_bdy.flatten()[:, None]
    y_bdy = y_bdy.flatten()[:, None]
    z_bdy = z_bdy.flatten()[:, None]
    u_bdy = u_bdy.flatten()[:, None]
    v_bdy = v_bdy.flatten()[:, None]
    w_bdy = w_bdy.flatten()[:, None]
    p_bdy = p_bdy.flatten()[:, None]
    f_bdy = f_bdy.flatten()[:, None]

    # load equations data
    x_eqns = np.array(pd.read_csv(file_dir + 'x_eqns.csv', header=None), dtype=np.float32)
    y_eqns = np.array(pd.read_csv(file_dir + 'y_eqns.csv', header=None), dtype=np.float32)
    z_eqns = np.array(pd.read_csv(file_dir + 'z_eqns.csv', header=None), dtype=np.float32)

    N_eqns = x_eqns.shape[0]

    t_eqns = np.tile(t, (1, N_eqns)).T
    x_eqns = np.tile(x_eqns, (1, T))
    y_eqns = np.tile(y_eqns, (1, T))
    z_eqns = np.tile(z_eqns, (1, T))

    t_eqns = t_eqns.flatten()[:, None]
    x_eqns = x_eqns.flatten()[:, None]
    y_eqns = y_eqns.flatten()[:, None]
    z_eqns = z_eqns.flatten()[:, None]

    # training
    model = chain_style_pinn(t_obs, x_obs, y_obs, z_obs, u_obs, v_obs, w_obs,
                             t_bdy, x_bdy, y_bdy, z_bdy, u_bdy, v_bdy, w_bdy, p_bdy, f_bdy,
                             t_eqns, x_eqns, y_eqns, z_eqns,
                             layers, batch_size)

    model.train(total_iteration=2e5, learning_rate=1e-3)
