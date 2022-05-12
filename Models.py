from __future__ import absolute_import
from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.weight_size = eval(args.layer_size)
        self.num_layer = len(self.weight_size)

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")  # (b, 1)
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # (num_users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # (num_items, embedding_size)
            self.h = tf.Variable(tf.random_uniform([self.weight_size[-1], 1], minval=-tf.sqrt(3 / self.weight_size[-1]),
                                                   maxval=tf.sqrt(3 / self.weight_size[-1])), name='h')  # (W[-1], 1)
            self.W, self.b = {}, {}
            self.weight_sizes = [2 * self.embedding_size] + self.weight_size
            for i in range(self.num_layer):
                self.W[i] = tf.Variable(tf.random_uniform(shape=[self.weight_sizes[i], self.weight_sizes[i + 1]],
                                                          minval=-tf.sqrt(
                                                              6 / (self.weight_sizes[i] + self.weight_sizes[i + 1])),
                                                          maxval=tf.sqrt(
                                                              6 / (self.weight_sizes[i] + self.weight_sizes[i + 1]))),
                                        name='W' + str(i), dtype=tf.float32)  # (2*embed_size, W[1]) (w[i],w[i+1])
                self.b[i] = tf.Variable(tf.zeros([1, self.weight_sizes[i + 1]]), dtype=tf.float32,
                                        name='b' + str(i))  # (1, W[i+1])

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input),
                                             1)  # (b, embedding_size)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)

            self.z = []
            z_temp = tf.concat([self.embedding_p, self.embedding_q], 1)  # (b, 2*embed_size)
            self.z.append(z_temp)

            for i in range(self.num_layer):
                z_temp = tf.nn.relu(tf.matmul(self.z[i], self.W[i]) + self.b[
                    i])  # (b, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (b, W[i+1])
                self.z.append(z_temp)
            return tf.sigmoid(tf.matmul(z_temp, self.h), name = 'output')  # (b, W[-1]) * (W[-1], 1) => (b, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(
                                tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            if self.loss_func == "logloss":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self.saver = tf.train.Saver() 


class GMF_controlled:
    def __init__(self, num_users, num_items, args, use_pretrain, load='ipv'):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        self.frozen_ui = args.frozen
        self.use_pretrain = use_pretrain
        self.model = args.model

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            '''
            f = open('./Param/param_beibei-cart' + "_P.txt", 'r')
            params = np.loadtxt(f)
            f.close()
            if 'user' in self.frozen:
                print 'frozen user'
                self.embedding_P = tf.Variable(
                    params, name='embedding_P', dtype=tf.float32, trainable=False)
            else:
                self.embedding_P = tf.Variable(params, name='embedding_P', dtype=tf.float32)

            f = open('./Param/param_beibei-cart' + "_Q.txt", 'r')
            params = np.loadtxt(f)
            f.close()
            if 'item' in self.frozen:
                print 'frozen item'
                self.embedding_Q = tf.Variable(
                    params, name='embedding_Q', dtype=tf.float32, trainable=False)
            else:
                self.embedding_Q = tf.Variable(
                    params, name='embedding_Q', dtype=tf.float32)
            '''
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32, trainable='user' not in self.frozen_ui)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32, trainable='item' not in self.frozen_ui)  # (items, embedding_size)
            # self.h = tf.Variable(tf.ones([self.embedding_size, 1]), name='h', dtype=tf.float32)  #how to initialize it  (embedding_size, 1)
            self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(3 / self.embedding_size),
                                                   maxval=tf.sqrt(3 / self.embedding_size)), name='h')
            self.variables_name = 'PQh'

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            return tf.sigmoid(
                tf.matmul(self.embedding_p * self.embedding_q, self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            # self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            if self.loss_func == "logloss":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self.saver = tf.train.Saver() 


class GMF_Model:
    def __init__(self, num_users, num_items, args, use_pretrain, load='ipv'):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        self.frozen_ui = args.frozen
        self.use_pretrain = use_pretrain
        self.add_fc = args.add_fc

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32, trainable='user' not in self.frozen_ui)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32, trainable='item' not in self.frozen_ui)  # (items, embedding_size)
            # self.h = tf.Variable(tf.ones([self.embedding_size, 1]), name='h', dtype=tf.float32)  #how to initialize it  (embedding_size, 1)
            self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(3 / self.embedding_size),
                                                   maxval=tf.sqrt(3 / self.embedding_size)), name='h')
            # if self.add_fc:
            # self.fc = tf.Variable(tf.random_uniform([self.embedding_size, self.embedding_size], minval=-tf.sqrt(3 / self.embedding_size),
            #                                    maxval=tf.sqrt(3 / self.embedding_size)), name='fc')
            # self.fc = tf.layers.dense(self.embedding_size, self.embedding_size)
            self.variables_name = 'PQh'

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            print 'item' in self.add_fc
            if 'user' in self.add_fc:
                print 'user'
                return tf.sigmoid(
                    tf.matmul(tf.layers.dense(self.embedding_p, self.embedding_size) * self.embedding_q,
                              self.h))  # (b, embedding_size) * (embedding_size, 1)
            elif 'item' in self.add_fc:
                print 'item'
                return tf.sigmoid(
                    tf.matmul(self.embedding_p * tf.layers.dense(self.embedding_q, self.embedding_size),
                              self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            # self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            if self.loss_func == "logloss":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self.saver = tf.train.Saver() 


class pure_GMF:
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]       
        self.loss_func = args.loss_func
        self.opt = args.optimizer

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
            # self.h = tf.Variable(tf.ones([self.embedding_size, 1]), name='h', dtype=tf.float32)  #how to initialize it  (embedding_size, 1)
            self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(3 / self.embedding_size),
                                                   maxval=tf.sqrt(3 / self.embedding_size)), name='h')

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            return tf.sigmoid(
                tf.matmul(self.embedding_p * self.embedding_q, self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                self.loss_no_reg = tf.losses.log_loss(self.labels, self.output)
                self.loss_reg = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                                self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class pure_MLP:
    def __init__(self, num_users, num_items, args, use_pretrain=False, saver=None, loader=None):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        self.frozen_ui = args.frozen
        self.use_pretrain = use_pretrain
        self.layer_num = args.layer_num
        self.opt = args.optimizer

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
            self.h = tf.Variable(
                tf.random_uniform([2 * int(self.embedding_size/(2**self.layer_num)), 1], minval=-tf.sqrt(1 / self.embedding_size),
                                  maxval=tf.sqrt(1 / self.embedding_size)), name='h')

        with tf.name_scope("FC"):         

            if self.layer_num == 0:
                pass
            elif self.layer_num == 1:
                self.W_FC = tf.Variable(tf.random_uniform(shape=[2*self.embedding_size, self.embedding_size],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W_FC')
                self.b_FC = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b_FC')
            else:
                self.W_FC = []
                self.b_FC = []
                for i in range(self.layer_num):
                    input_size = int(2*self.embedding_size/(2**i))
                    output_size = int(2*self.embedding_size/(2**(i+1)))
                    self.W_FC.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W_FC_%d' %i))
                    self.b_FC.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b_FC_%d' %i))


    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            self.concat_vec = tf.concat([self.embedding_p, self.embedding_q], 1, name='concat_vec')
            
            if self.layer_num == 0:
                return tf.sigmoid(tf.matmul(self.concat_vec, self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

            elif self.layer_num == 1:
                fc = tf.nn.relu(tf.matmul(self.concat_vec, self.W_FC) + self.b_FC)
                return tf.sigmoid(tf.matmul(fc, self.h), name = 'output')
           
            else:
                fc = []
                for i in range(self.layer_num):
                    if i == 0:
                        fc.append(tf.nn.relu(tf.matmul(self.concat_vec, self.W_FC[i]) + self.b_FC[i]))
                    else:
                        fc.append(tf.nn.relu(tf.matmul(fc[i-1], self.W_FC[i]) + self.b_FC[i]))

                return tf.sigmoid(tf.matmul(fc[i], self.h), name = 'output')

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))


    # def _create_optimizer(self):
    #     with tf.name_scope("optimizer"):
    #         if self.loss_func == "logloss":
    #             if self.opt == 'Adam':
    #                 self.optimizer = tf.train.AdamOptimizer(
    #                     learning_rate = self.learning_rate).minimize(self.loss)
    #             elif self.opt == 'Adagrad':
    #                 self.optimizer = tf.train.AdagradOptimizer(
    #                     learning_rate = self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)  
    #         else:
    #             self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class pure_NCF:
    def __init__(self, num_users, num_items, args, use_pretrain=False, saver=None, loader=None):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        self.frozen_ui = args.frozen
        self.use_pretrain = use_pretrain
        self.layer_num = args.layer_num
        self.opt = args.optimizer

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
            self.h = tf.Variable(
                tf.random_uniform([3 * int(self.embedding_size/(2**self.layer_num)), 1], minval=-tf.sqrt(1 / self.embedding_size),
                                  maxval=tf.sqrt(1 / self.embedding_size)), name='h')

        with tf.name_scope("FC"):         

            if self.layer_num == 0:
                pass
            elif self.layer_num == 1:
                self.W_FC = tf.Variable(tf.random_uniform(shape=[3*self.embedding_size, int(3*self.embedding_size/2)],
                                                    minval=-tf.sqrt(1 / self.embedding_size),
                                                    maxval=tf.sqrt(1 / self.embedding_size)), name='W_FC')
                self.b_FC = tf.Variable(tf.zeros([1, int(3*self.embedding_size/2)]), dtype=tf.float32, name='b_FC')
            else:
                self.W_FC = []
                self.b_FC = []
                for i in range(self.layer_num):
                    input_size = int(3*self.embedding_size/(2**i))
                    output_size = int(3*self.embedding_size/(2**(i+1)))
                    self.W_FC.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W_FC_%d' %i))
                    self.b_FC.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b_FC_%d' %i))


    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            self.concat_vec = tf.concat([self.embedding_p, self.embedding_q, self.embedding_p * self.embedding_q], 1, name='concat_vec')
            if self.layer_num == 0:
                return tf.sigmoid(tf.matmul(self.concat_vec, self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

            elif self.layer_num == 1:
                fc = tf.nn.relu(tf.matmul(self.concat_vec, self.W_FC) + self.b_FC)
                return tf.sigmoid(tf.matmul(fc, self.h), name = 'output')
           
            else:
                fc = []
                for i in range(self.layer_num):
                    if i == 0:
                        fc.append(tf.nn.relu(tf.matmul(self.concat_vec, self.W_FC[i]) + self.b_FC[i]))
                    else:
                        fc.append(tf.nn.relu(tf.matmul(fc[i-1], self.W_FC[i]) + self.b_FC[i]))

                return tf.sigmoid(tf.matmul(fc[i], self.h), name = 'output')

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(self.item_input[:, 0])
                self.output_neg = self._create_inference(self.item_input[:, -1])
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                    tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))

    # def _create_optimizer(self):
    #     with tf.name_scope("optimizer"):
    #         if self.loss_func == "logloss":
    #             if self.opt == 'Adam':
    #                 self.optimizer = tf.train.AdamOptimizer(
    #                     learning_rate = self.learning_rate).minimize(self.loss)
    #             elif self.opt == 'Adagrad':
    #                 self.optimizer = tf.train.AdagradOptimizer(
    #                     learning_rate = self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)  
    #         else:
    #             self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class Multi_GMF():
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        loss_coefficient = eval(args.loss_coefficient)
        self.cart_loss_coefficient = loss_coefficient[0]
        self.buy_loss_coefficient = loss_coefficient[1]
        self.ipv_loss_coefficient = 1 - self.cart_loss_coefficient - self.buy_loss_coefficient
        self.opt = args.optimizer
        self.b_num = args.b_num
        self.b_2_type = args.b_2_type

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape = [None, None], name = 'item_input')
            self.labels_ipv = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_ipv')
            self.labels_cart = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_cart')
            self.labels_buy = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_buy')

    def _create_variables(self):
        with tf.name_scope('embedding'):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape = [self.num_users, self.embedding_size], 
                    mean = 0.0, stddev = 0.01), name = 'embedding_P', dtype = tf.float32)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape = [self.num_items, self.embedding_size],
                    mean = 0.0, stddev = 0.01), name = 'embedding_Q', dtype = tf.float32)
            self.bias = tf.Variable(
                tf.zeros([self.num_items, 1]), 
                name = 'bias', dtype = tf.float32)

            # [E, E]
            self.W1 = tf.Variable(
                tf.random_uniform(shape = [self.embedding_size, self.embedding_size],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'W1', dtype = tf.float32))
            self.W2 = tf.Variable(
                tf.random_uniform(shape = [self.embedding_size, self.embedding_size],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'W2', dtype = tf.float32))
            self.W3 = tf.Variable(
                tf.random_uniform(shape = [self.embedding_size, self.embedding_size],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'W3', dtype = tf.float32))

            # [E, 1]
            self.h_1 = tf.Variable(
                tf.random_uniform([self.embedding_size, 1],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'h_1', dtype = tf.float32))
            self.h_2 = tf.Variable(
                tf.random_uniform([self.embedding_size, 1],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'h_2', dtype = tf.float32))
            self.h_3 = tf.Variable(
                tf.random_uniform([self.embedding_size, 1],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'h_3', dtype = tf.float32))


    def _create_inference(self):
        with tf.name_scope('inference'):
            # [B, E]
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            z = embedding_p * embedding_q
            # [B, 1]
            b = tf.reduce_sum(tf.nn.embedding_lookup(self.bias, self.item_input), 1)            
            
            # view, cart, buy
            if self.b_num == 3:
                # predict ipv
                output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1)), self.h_1) + b
                # predict cart
                temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2)), self.h_2)
                output_cart = (temp_cart + output_ipv) / 2 + b
                # predict buy
                temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W3)), self.h_3)
                output_buy = (temp_buy + output_cart) / 2 + b

                return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                        tf.sigmoid(output_cart, name = 'score_cart'), 
                        tf.sigmoid(output_buy, name = 'score_buy'))
           
            else:
                if self.b_2_type == 'cb':
                    # predict cart
                    output_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1)), self.h_1) + b
                    # predict buy
                    temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2)), self.h_2)
                    output_buy = (temp_buy + output_cart) / 2 + b                

                    return (tf.sigmoid(output_cart, name = 'score_cart'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))
                
                elif self.b_2_type == 'vc':
                    # predict view
                    output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1)), self.h_1) + b
                    # predict cart
                    temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2)), self.h_2)
                    output_cart = (temp_cart + output_ipv) / 2 + b                

                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_cart, name = 'score_cart'))    

                else:                
                    # predict view
                    output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1)), self.h_1) + b
                    # predict buy
                    temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2)), self.h_2)
                    output_buy = (temp_buy + output_ipv) / 2 + b                

                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))    


    def _create_loss(self):
        with tf.name_scope('loss'):
            if self.loss_func == 'logloss':
                self.loss_reg_2 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                                  self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                                  self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1)) + \
                                  self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2)) 


                self.loss_reg_3 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                                  self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                                  self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1)) + \
                                  self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2)) + \
                                  self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3)) 


                if self.b_num == 3:
                    self.score_ipv, self.score_cart, self.score_buy = self._create_inference()
                    self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                    self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                    self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                    self.loss = self.ipv_loss_coefficient * self.loss_ipv + \
                                self.cart_loss_coefficient * self.loss_cart + \
                                self.buy_loss_coefficient * self.loss_buy + self.loss_reg_3
            
                else:
                    if self.b_2_type == 'cb':
                        self.score_cart, self.score_buy = self._create_inference()
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_cart + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg_2
                    elif self.b_2_type == 'vb':
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg_2
                    else:
                        self.score_ipv, self.score_cart = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_cart + self.loss_reg_2
            else:
                pass

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class Multi_MLP():
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        loss_coefficient = eval(args.loss_coefficient)
        self.cart_loss_coefficient = loss_coefficient[0]
        self.buy_loss_coefficient = loss_coefficient[1]
        self.ipv_loss_coefficient = 1 - self.cart_loss_coefficient - self.buy_loss_coefficient
        self.opt = args.optimizer
        self.layer_num = args.layer_num
        self.b_num = args.b_num
        self.b_2_type = args.b_2_type

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape = [None, None], name = 'item_input')
            self.labels_ipv = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_ipv')
            self.labels_cart = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_cart')
            self.labels_buy = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_buy')

    def _create_variables(self):

        with tf.name_scope('shared_embedding'):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape = [self.num_users, self.embedding_size], 
                    mean = 0.0, stddev = 0.01), name = 'embedding_P', dtype = tf.float32)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape = [self.num_items, self.embedding_size],
                    mean = 0.0, stddev = 0.01), name = 'embedding_Q', dtype = tf.float32)

        with tf.name_scope('shared_bias'):    
            self.bias = tf.Variable(
                tf.zeros([self.num_items, 1]), 
                name = 'bias', dtype = tf.float32)

        with tf.name_scope('W_b_h'):          
            # h-vector
            h_size = 2 * int(self.embedding_size/(2**self.layer_num))
            self.h_1 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_1')
            self.h_2 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_2')
            self.h_3 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_3')                                         

            if self.layer_num == 0:
                pass         
            elif self.layer_num == 1:
                # view specific 
                self.W1 = tf.Variable(tf.random_uniform(shape=[2*self.embedding_size, self.embedding_size],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W1')
                self.b1 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b1')
                # add cart specific
                self.W2 = tf.Variable(tf.random_uniform(shape=[2*self.embedding_size, self.embedding_size],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W2')
                self.b2 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b2')
                # buy specific
                self.W3 = tf.Variable(tf.random_uniform(shape=[2*self.embedding_size, self.embedding_size],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W3')
                self.b3 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b3')

            else:
                self.W1, self.b1 = [], []
                self.W2, self.b2 = [], []
                self.W3, self.b3 = [], []

                for i in range(self.layer_num):
                    input_size = int(2*self.embedding_size/(2**i))
                    output_size = int(2*self.embedding_size/(2**(i+1)))
                    self.W1.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W1_%d' %i))
                    self.b1.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b1_%d' %i))
                    self.W2.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W2_%d' %i))
                    self.b2.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b2_%d' %i))
                    self.W3.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W3_%d' %i))
                    self.b3.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b3_%d' %i))


    def _create_inference(self):

        with tf.name_scope('inference'):
            # [B, 1] item-popularity
            b = tf.reduce_sum(tf.nn.embedding_lookup(self.bias, self.item_input), 1)
            # [B, E] shared embeddings
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            z =  tf.concat([embedding_p, embedding_q], 1, name='z')
            
            if self.layer_num == 0:
                pass

            elif self.layer_num == 1:
                if self.b_num == 3:
                    # predict ipv
                    output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                    # predict cart
                    temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                    output_cart = (temp_cart + output_ipv) / 2 + b
                    # predict buy
                    temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W3) + self.b3), self.h_3)
                    output_buy = (temp_buy + output_cart) / 2 + b

                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_cart, name = 'score_cart'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))
                
                else:
                    if self.b_2_type == 'cb':
                        # predict cart
                        output_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_buy = (temp_buy + output_cart) / 2 + b

                        return (tf.sigmoid(output_cart, name = 'score_cart'), 
                                tf.sigmoid(output_buy, name = 'score_buy')) 

                    elif self.b_2_type == 'vc':
                        # predict ipv
                        output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict cart
                        temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_cart = (temp_cart + output_ipv) / 2 + b          

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_cart, name = 'score_cart'))   

                    else:
                        # predict ipv
                        output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_buy = (temp_buy + output_ipv) / 2 + b           

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_buy, name = 'score_buy'))    
          

            else:
                fc_1, fc_2, fc_3 = [], [], []
                for i in range(self.layer_num):
                    if i == 0:
                        fc_1.append(tf.nn.relu(tf.matmul(z, self.W1[i]) + self.b1[i]))
                        fc_2.append(tf.nn.relu(tf.matmul(z, self.W2[i]) + self.b2[i]))
                        fc_3.append(tf.nn.relu(tf.matmul(z, self.W3[i]) + self.b3[i]))

                    else:
                        fc_1.append(tf.nn.relu(tf.matmul(fc_1[i-1], self.W1[i]) + self.b1[i]))
                        fc_2.append(tf.nn.relu(tf.matmul(fc_2[i-1], self.W2[i]) + self.b2[i]))
                        fc_3.append(tf.nn.relu(tf.matmul(fc_3[i-1], self.W3[i]) + self.b3[i]))


                if self.b_num == 3:
                    # predict ipv
                    output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                    # predict cart
                    temp_cart = tf.matmul(fc_2[i], self.h_2)
                    output_cart = (temp_cart + output_ipv) / 2 + b
                    # predict buy
                    temp_buy = tf.matmul(fc_3[i], self.h_3)
                    output_buy = (temp_buy + output_cart) / 2 + b

                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_cart, name = 'score_cart'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))
                
                else:
                    if self.b_2_type == 'cb':
                        # predict cart
                        output_cart = tf.matmul(fc_1[i], self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(fc_2[i], self.h_2)
                        output_buy = (temp_buy + output_cart) / 2 + b

                        return (tf.sigmoid(output_cart, name = 'score_cart'), 
                                tf.sigmoid(output_buy, name = 'score_buy')) 

                    elif self.b_2_type == 'vc':
                        # predict ipv
                        output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                        # predict cart
                        temp_cart = tf.matmul(fc_2[i], self.h_2)
                        output_cart = (temp_buy + output_ipv) / 2 + b     

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_cart, name = 'score_cart'))   

                    else:
                        # predict ipv
                        output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(fc_2[i], self.h_2)
                        output_buy = (temp_buy + output_ipv) / 2 + b            

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_buy, name = 'score_buy'))    


    def _create_loss(self):
        with tf.name_scope('loss'):
            if self.loss_func == 'logloss':
                loss_em = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                          self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                loss_W1 = 0
                loss_W2 = 0
                loss_W3 = 0
                if self.layer_num == 1:
                    loss_W1 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1))
                    loss_W2 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2))
                    if self.b_num == 3:
                        loss_W3 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3))

                else:
                    for i in range(len(self.W1)):
                        loss_W1 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1[i]))
                        loss_W2 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2[i]))
                        if self.b_num == 3:
                            loss_W3 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3[i]))

                self.loss_reg = loss_em + loss_W1 + loss_W2 + loss_W3


                if self.b_num == 3:
                    self.score_ipv, self.score_cart, self.score_buy = self._create_inference()
                    self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                    self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                    self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                    self.loss = self.ipv_loss_coefficient * self.loss_ipv + \
                                self.cart_loss_coefficient * self.loss_cart + \
                                self.buy_loss_coefficient * self.loss_buy + self.loss_reg
            
                else:
                    if self.b_2_type == 'cb':
                        self.score_cart, self.score_buy = self._create_inference()
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_cart + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg
                    elif self.b_2_type == 'vb':
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg
                    else:
                        self.score_ipv, self.score_cart = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_cart + self.loss_reg
            else:
                pass


    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()



class Multi_NCF():
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        loss_coefficient = eval(args.loss_coefficient)
        self.cart_loss_coefficient = loss_coefficient[0]
        self.buy_loss_coefficient = loss_coefficient[1]
        self.ipv_loss_coefficient = 1 - self.cart_loss_coefficient - self.buy_loss_coefficient
        self.opt = args.optimizer
        self.layer_num = args.layer_num
        self.b_num = args.b_num
        self.b_2_type = args.b_2_type


    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape = [None, None], name = 'item_input')
            self.labels_ipv = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_ipv')
            self.labels_cart = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_cart')
            self.labels_buy = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels_buy')

    def _create_variables(self):

        with tf.name_scope('shared_embedding'):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape = [self.num_users, self.embedding_size], 
                    mean = 0.0, stddev = 0.01), name = 'embedding_P', dtype = tf.float32)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape = [self.num_items, self.embedding_size],
                    mean = 0.0, stddev = 0.01), name = 'embedding_Q', dtype = tf.float32)
 

        with tf.name_scope('shared_bias'):    
            self.bias = tf.Variable(
                tf.zeros([self.num_items, 1]), 
                name = 'bias', dtype = tf.float32)


        with tf.name_scope('W_b_h'):
            
            # h-vector
            h_size = 3 * int(self.embedding_size/(2**self.layer_num))
            self.h_1 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_1')
            self.h_2 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_2')
            self.h_3 = tf.Variable(
            tf.random_uniform([h_size, 1], minval=-tf.sqrt(3 / h_size),
                              maxval=tf.sqrt(3 / h_size)), name='h_3')                                         

            if self.layer_num == 0:
                pass
            
            elif self.layer_num == 1:
                # view specific 
                self.W1 = tf.Variable(tf.random_uniform(shape=[3*self.embedding_size, int(3*self.embedding_size/2)],
                                    minval=-tf.sqrt(1 / self.embedding_size),
                                    maxval=tf.sqrt(1 / self.embedding_size)), name='W1')
                self.b1 = tf.Variable(tf.zeros([1, int(3*self.embedding_size/2)]), dtype=tf.float32, name='b1')

                # add cart specific
                self.W2 = tf.Variable(tf.random_uniform(shape=[3*self.embedding_size, int(3*self.embedding_size/2)],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W2')
                self.b2 = tf.Variable(tf.zeros([1, int(3*self.embedding_size/2)]), dtype=tf.float32, name='b2')

                # buy specific
                self.W3 = tf.Variable(tf.random_uniform(shape=[3*self.embedding_size, int(3*self.embedding_size/2)],
                                    minval=-tf.sqrt(3 / (2*self.embedding_size)),
                                    maxval=tf.sqrt(3 / (2*self.embedding_size))), name='W3')
                self.b3 = tf.Variable(tf.zeros([1, int(3*self.embedding_size/2)]), dtype=tf.float32, name='b3')

            else:
                self.W1, self.b1 = [], []
                self.W2, self.b2 = [], []
                self.W3, self.b3 = [], []

                for i in range(self.layer_num):
                    input_size = int(3*self.embedding_size/(2**i))
                    output_size = int(3*self.embedding_size/(2**(i+1)))
                    self.W1.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W1_%d' %i))
                    self.b1.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b1_%d' %i))
                    self.W2.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W2_%d' %i))
                    self.b2.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b2_%d' %i))
                    self.W3.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-tf.sqrt(3 / input_size),
                                                    maxval=tf.sqrt(3 / input_size)), name='W3_%d' %i))
                    self.b3.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b3_%d' %i))


    def _create_inference(self):

        with tf.name_scope('inference'):
            # [B, 1] item-popularity
            b = tf.reduce_sum(tf.nn.embedding_lookup(self.bias, self.item_input), 1)

            # [B, E] shared embeddings
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)

            # [B, 3E]
            z =  tf.concat([embedding_p, embedding_q, embedding_p * embedding_q], 1, name='z')
            
            if self.layer_num == 0:
                pass

            elif self.layer_num == 1:
                if self.b_num == 3:
                    # predict ipv
                    output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                    # predict cart
                    temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                    output_cart = (temp_cart + output_ipv) / 2 + b
                    # predict buy
                    temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W3) + self.b3), self.h_3)
                    output_buy = (temp_buy + output_cart) / 2 + b
                    
                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_cart, name = 'score_cart'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))
                else:
                    if self.b_2_type == 'cb':
                        # predict cart
                        output_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_buy = (temp_buy + output_cart) / 2 + b

                        return (tf.sigmoid(output_cart, name = 'score_cart'), 
                                tf.sigmoid(output_buy, name = 'score_buy')) 

                    elif self.b_2_type == 'vc':
                        # predict ipv
                        output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict cart
                        temp_cart = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_cart = (temp_cart + output_ipv) / 2 + b          

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_cart, name = 'score_cart'))   

                    else:
                        # predict ipv
                        output_ipv = tf.matmul(tf.nn.relu(tf.matmul(z, self.W1) + self.b1), self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(tf.nn.relu(tf.matmul(z, self.W2) + self.b2), self.h_2)
                        output_buy = (temp_buy + output_ipv) / 2 + b           

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_buy, name = 'score_buy'))    

            else:
                fc_1, fc_2, fc_3 = [], [], []
                for i in range(self.layer_num):
                    if i == 0:
                        fc_1.append(tf.nn.relu(tf.matmul(z, self.W1[i]) + self.b1[i]))
                        fc_2.append(tf.nn.relu(tf.matmul(z, self.W2[i]) + self.b2[i]))
                        fc_3.append(tf.nn.relu(tf.matmul(z, self.W3[i]) + self.b3[i]))

                    else:
                        fc_1.append(tf.nn.relu(tf.matmul(fc_1[i-1], self.W1[i]) + self.b1[i]))
                        fc_2.append(tf.nn.relu(tf.matmul(fc_2[i-1], self.W2[i]) + self.b2[i]))
                        fc_3.append(tf.nn.relu(tf.matmul(fc_3[i-1], self.W3[i]) + self.b3[i]))

                if self.b_num == 3:
                    # predict ipv
                    output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                    # predict cart
                    temp_cart = tf.matmul(fc_2[i], self.h_2)
                    output_cart = (temp_cart + output_ipv) / 2 + b
                    # predict buy
                    temp_buy = tf.matmul(fc_3[i], self.h_3)
                    output_buy = (temp_buy + output_cart) / 2 + b

                    return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                            tf.sigmoid(output_cart, name = 'score_cart'), 
                            tf.sigmoid(output_buy, name = 'score_buy'))
                else:
                    if self.b_2_type == 'cb':
                        # predict cart
                        output_cart = tf.matmul(fc_1[i], self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(fc_2[i], self.h_2)
                        output_buy = (temp_buy + output_cart) / 2 + b

                        return (tf.sigmoid(output_cart, name = 'score_cart'), 
                                tf.sigmoid(output_buy, name = 'score_buy')) 

                    elif self.b_2_type == 'vc':
                        # predict ipv
                        output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                        # predict cart
                        temp_cart = tf.matmul(fc_2[i], self.h_2)
                        output_cart = (temp_buy + output_ipv) / 2 + b     

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_cart, name = 'score_cart'))   

                    else:
                        # predict ipv
                        output_ipv = tf.matmul(fc_1[i], self.h_1) + b
                        # predict buy
                        temp_buy = tf.matmul(fc_2[i], self.h_2)
                        output_buy = (temp_buy + output_ipv) / 2 + b            

                        return (tf.sigmoid(output_ipv, name = 'score_ipv'), 
                                tf.sigmoid(output_buy, name = 'score_buy'))    


    def _create_loss(self):
        with tf.name_scope('loss'):
            if self.loss_func == 'logloss':
                loss_em = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                          self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                loss_W1 = 0
                loss_W2 = 0
                loss_W3 = 0
                if self.layer_num == 1:
                    loss_W1 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1))
                    loss_W2 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2))
                    if self.b_num == 3:
                        loss_W3 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3))

                else:
                    for i in range(len(self.W1)):
                        loss_W1 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1[i]))
                        loss_W2 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2[i]))
                        if self.b_num == 3:
                            loss_W3 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3[i]))

                self.loss_reg = loss_em + loss_W1 + loss_W2 + loss_W3


                if self.b_num == 3:
                    self.score_ipv, self.score_cart, self.score_buy = self._create_inference()
                    self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                    self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                    self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                    self.loss = self.ipv_loss_coefficient * self.loss_ipv + \
                                self.cart_loss_coefficient * self.loss_cart + \
                                self.buy_loss_coefficient * self.loss_buy + self.loss_reg
            
                else:
                    if self.b_2_type == 'cb':
                        self.score_cart, self.score_buy = self._create_inference()
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_cart + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg
                    elif self.b_2_type == 'vb':
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_buy + self.loss_reg
                    else:
                        self.score_ipv, self.score_cart = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss = self.cart_loss_coefficient * self.loss_ipv + \
                                    self.buy_loss_coefficient * self.loss_cart + self.loss_reg
            else:
                pass


    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class BPR:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.dropout = args.dropout


    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
    
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], 
                    mean=0.0, stddev=0.01), name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], 
                    mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)
            
            self.h = tf.Variable(
                tf.random_uniform([self.embedding_size, 1],
                    minval = -tf.sqrt(3 / self.embedding_size), maxval = tf.sqrt(3 / self.embedding_size),
                    name = 'h', dtype = tf.float32))

    def _create_inference(self, item_input, name):
        with tf.name_scope("inference"):
            embedding_p = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1), self.dropout)
            embedding_q = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1), self.dropout) 
            #(b, embedding_size)
            
            return tf.matmul(embedding_p * embedding_q, self.h, name = name)  #(b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.output = self._create_inference(self.item_input, 'output')
            self.output_neg = self._create_inference(self.item_input_neg, 'output_neg')
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) + \
                self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            
            # self.opt_loss = self.loss + self.lambda_bilinear * tf.reduce_sum(tf.square(self.p1)) \
            #                     + self.lambda_bilinear * tf.reduce_sum(tf.square(self.p2)) \
            #                     + self.gamma_bilinear * tf.reduce_sum(tf.square(self.q1))
        

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class FISM:

    def __init__(self, num_items, num_users, max_rate, args):
        self.num_items = num_items
        self.num_users = num_users
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.alpha = args.alpha
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.max_rate = max_rate


    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            # item_rate: all rated items for a user of except for item_input
            # item_num: len(item_rate)
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = 'user_input')
            self.item_input = tf.placeholder(tf.int32, shape = [None, 1], name = 'item_input')
            self.item_rate = tf.placeholder(tf.int32, shape = [None, None], name = 'item_rate')  # [B, max_rate]
            self.item_num = tf.placeholder(tf.float32, shape = [None, 1], name = 'item_num')  
            self.labels = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels')


    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape = [self.num_items, self.embedding_size], 
                    mean = 0.0, stddev = 0.01), name = 'embedding_P', dtype = tf.float32)           
            self.embedding_p = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='embedding_p' )
            self.embedding_P_long = tf.concat([self.embedding_P, self.embedding_p], 0, name='embedding_P_long')
                        
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape = [self.num_items, self.embedding_size], 
                    mean = 0.0, stddev = 0.01), name = 'embedding_Q', dtype = tf.float32)

            self.bias_u = tf.Variable(tf.zeros([self.num_users, 1]), name = 'bias_u', dtype = tf.float32)      
            self.bias_i = tf.Variable(tf.zeros([self.num_items, 1]), name = 'bias_i', dtype = tf.float32)    


    def _create_inference(self):
        with tf.name_scope('inference'):
            for i in range(self.max_rate):
                p_j = tf.slice(self.item_rate, [0, i], [-1, 1])  # [B, 1] choose j from max_rate
                p_j_e = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_long, p_j), 1)  # [B, E] 
                if i == 0:
                    p_j_sum = p_j_e
                else:
                    p_j_sum = p_j_sum + p_j_e
     
            q_i_e = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)  # [B, E]
            item_sim = q_i_e * p_j_sum
            
            # [B, 1]
            b_i = tf.reduce_sum(tf.nn.embedding_lookup(self.bias_i, self.item_input), 1)
            b_u = tf.reduce_sum(tf.nn.embedding_lookup(self.bias_u, self.user_input), 1)
            # predict ipv
            output = b_i + b_u + tf.pow(self.item_num, -tf.constant(self.alpha, tf.float32)) * item_sim
            return tf.sigmoid(output, name = 'output')  # [B, 1]


    def _create_loss(self):
        with tf.name_scope('loss'):
            self.output = self._create_inference()
            self.loss = tf.pow((self.output - self.labels), tf.constant(2, tf.float32)) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))               

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class NCF_FC:
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_P', dtype=tf.float32)

            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_Q', dtype=tf.float32)

            self.h_1 = tf.Variable(tf.random_uniform([self.embedding_size*3, 1], 
                                                      minval=-tf.sqrt(1 / self.embedding_size), 
                                                      maxval=tf.sqrt(1 / self.embedding_size)), 
                                   name='h_1',dtype = tf.float32)

            self.h_2 = tf.Variable(tf.random_uniform([self.embedding_size*3, 1], 
                                                      minval=-tf.sqrt(1 / self.embedding_size), 
                                                      maxval=tf.sqrt(1 / self.embedding_size)), 
                                   name='h_2',dtype = tf.float32)

            self.h_3 = tf.Variable(tf.random_uniform([self.embedding_size*3, 1], 
                                                      minval=-tf.sqrt(1 / self.embedding_size), 
                                                      maxval=tf.sqrt(1 / self.embedding_size)), 
                                   name='h_3',dtype = tf.float32)

            # Add multi-layer
            self.W1 = tf.Variable(tf.random_uniform(shape=[3 * self.embedding_size, 3 * self.embedding_size],
                                                    minval=-tf.sqrt(1 / self.embedding_size),
                                                    maxval=tf.sqrt(1 / self.embedding_size)), name='W1')

            self.b1 = tf.Variable(tf.zeros([1, 3 * self.embedding_size]), dtype=tf.float32, name='b1')

            self.W2 = tf.Variable(tf.random_uniform(shape=[3 * self.embedding_size, 3 * self.embedding_size],
                                                    minval=-tf.sqrt(1 / self.embedding_size),
                                                    maxval=tf.sqrt(1 / self.embedding_size)), name='W2')

            self.b2 = tf.Variable(tf.zeros([1, 3 * self.embedding_size]), dtype=tf.float32, name='b2')

    def _create_inference(self):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)  # (b, embedding_size)

            z = tf.concat([embedding_p, embedding_q, embedding_p * embedding_q], 1)

            output1 = tf.sigmoid(tf.matmul(z, self.h_1), name = 'output1')  
           
            fc1 = tf.nn.relu(tf.matmul(z, self.W1) + self.b1)
            output2 = tf.sigmoid(tf.matmul(fc1, self.h_2), name = 'output2')
            
            fc2 = tf.nn.relu(tf.matmul(fc1, self.W2) + self.b2)
            output3 = tf.sigmoid(tf.matmul(fc2, self.h_3), name = 'output3')

            return output1, output2, output3

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output1, self.output2, self.output3 = self._create_inference()
                self.loss1 = tf.losses.log_loss(self.labels, self.output1) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                self.loss2 = tf.losses.log_loss(self.labels, self.output2) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                self.loss3 = tf.losses.log_loss(self.labels, self.output3) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                pass

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


class GMF_FC:
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (b,1)

    def _create_variables(self):
        with tf.name_scope("embedding"):

            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)

            self.h_1 = tf.Variable(tf.random_uniform([self.embedding_size, 1], 
                                                      minval=-tf.sqrt(3 / self.embedding_size), 
                                                      maxval=tf.sqrt(3 / self.embedding_size)), 
                                   name='h_1',dtype = tf.float32)

            self.h_2 = tf.Variable(tf.random_uniform([self.embedding_size, 1], 
                                                      minval=-tf.sqrt(3 / self.embedding_size), 
                                                      maxval=tf.sqrt(3 / self.embedding_size)), 
                                   name='h_2',dtype = tf.float32)

            self.h_3 = tf.Variable(tf.random_uniform([self.embedding_size, 1], 
                                                      minval=-tf.sqrt(3 / self.embedding_size), 
                                                      maxval=tf.sqrt(3 / self.embedding_size)), 
                                   name='h_3',dtype = tf.float32)
            # Add multi-layer
            self.W1 = tf.Variable(tf.random_uniform(shape=[self.embedding_size, self.embedding_size],
                                                    minval=-tf.sqrt(3 / self.embedding_size),
                                                    maxval=tf.sqrt(3 / self.embedding_size)), name='W1')

            self.b1 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b1')
            
            self.W2 = tf.Variable(tf.random_uniform(shape=[self.embedding_size, self.embedding_size],
                                                    minval=-tf.sqrt(3 / self.embedding_size),
                                                    maxval=tf.sqrt(3 / self.embedding_size)), name='W2')

            self.b2 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b2')

    def _create_inference(self):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            
            z = embedding_p * embedding_q

            output1 = tf.sigmoid(tf.matmul(z, self.h_1), name = 'output1')  

            fc1 = tf.nn.relu(tf.matmul(z, self.W1) + self.b1)
            output2 = tf.sigmoid(tf.matmul(fc1, self.h_2), name = 'output2')

            fc2 = tf.nn.relu(tf.matmul(fc1, self.W2) + self.b2)
            output3 = tf.sigmoid(tf.matmul(fc2, self.h_3), name = 'output3')
            
            return output1, output2, output3


    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output1, self.output2, self.output3 = self._create_inference()
                self.loss1 = tf.losses.log_loss(self.labels, self.output1) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                self.loss2 = tf.losses.log_loss(self.labels, self.output2) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
                self.loss3 = tf.losses.log_loss(self.labels, self.output3) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            
            else:
                # self.output = self._create_inference(self.item_input[:, 0])
                # self.output_neg = self._create_inference(self.item_input[:, -1])
                # self.result = self.output - self.output_neg
                # self.loss = tf.reduce_sum(tf.sigmoid(self.result) + self.lambda_bilinear * tf.reduce_sum(
                #     tf.square(self.embedding_P)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)))
                pass

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()





