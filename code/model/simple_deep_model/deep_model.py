import tensorflow as tf
import code.feature_engineering
import numpy as np
from sklearn.model_selection import train_test_split


# a = tf.Variable([1, 2, 3])
# b = tf.Variable([1, 2, 5])
# weight = tf.Variable([1, 1, 0])
# loss = tf.losses.mean_squared_error(a, b, weights=weight)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(loss))
class Model:
    def __init__(self):
        self.input = None
        self.conv_output = None
        self.output = None

        self.sess = tf.Session()

    def convs(self):
        with tf.name_scope('conv_layers') as scope:
            with tf.name_scope("conv1") as scope:
                kernel = tf.Variable(tf.truncated_normal([4, 1, 1], dtype=tf.float32, stddev=1e-1), name='weights',
                                     trainable=True)
                conv = tf.nn.conv1d(self.input, kernel, 1, padding="SAME")
                self.conv1 = tf.nn.relu(conv, name=scope)
            with tf.name_scope("pool1") as scope:
                self.pool1 = tf.layers.max_pooling1d(self.conv1, pool_size=2, strides=2)

        self.conv_output = self.pool1

    def denses(self):
        with tf.name_scope('fc_layers') as scope:
            # fc1
            with tf.name_scope('fc1') as scope:
                shape = int(np.prod(self.conv_output.get_shape()[1:]))
                fc1w = tf.Variable(tf.truncated_normal([shape, 37],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
                fc1b = tf.Variable(tf.constant(0.1, shape=[37], dtype=tf.float32),
                                   trainable=True, name='biases')
                pool5_flat = tf.reshape(self.conv_output, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                self.fc1 = tf.nn.sigmoid(fc1l)
                # self.parameters += [fc1w, fc1b]

        self.output = self.fc1

    def build(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 300, 1])
        # self.output = tf.placeholder(tf.int64, shape=[None, 37])
        self.expected_output = tf.placeholder(tf.int64, shape=[None, 37])
        self.weights = tf.placeholder(tf.int64, shape=[None, 37])
        self.convs()
        self.denses()

        print(self.output.shape)
        self.losses = tf.losses.mean_squared_error(self.expected_output, self.output, weights=self.weights)

        self.train_step = tf.train.AdamOptimizer(6e-5).minimize(self.losses)

    def train(self):
        all_rna, all_label = code.feature_engineering.get_data_sep(data_path="../../../dataset/RNA_trainset/")
        all_rna = np.array(all_rna)
        all_label = np.array(all_label)
        all_label = np.reshape(all_label, [])
        X_train, X_test, y_train, y_test = train_test_split(all_rna, all_label, test_size=0.2, random_state=42)
        print(y_train.shape)
        print(y_train[0])

        print(y_train[0].shape)
        print(y_train)
        weights = np.sign(y_train + 1)

        self.build()
        self.sess.run(tf.global_variables_initializer())
        for step in range(3):
            loss = self.sess.run(fetches=[self.losses], feed_dict={self.input: X_train, self.expected_output: y_train,
                                                                   self.weights: weights})
            print(loss)


m = Model()
m.train()
