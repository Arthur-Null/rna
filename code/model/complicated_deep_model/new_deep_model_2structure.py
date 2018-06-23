"""sys.argv[1]: epoch
   sys.argv[2]: train_batch_size
   sys.argv[3]: test_batch_size
"""
import sys

sys.path.insert(0, "../../../")
import tensorflow as tf
import numpy as np
import os
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from code.feature_engineering import get_data
from code.feature_engineering import aucs
from code.feature_engineering import get_data_sep

# data, label = get_data(data_path="../../../dataset/trainset/")
# print(len(label))
#
# rnas = []
# for rna in data:
#     encoder = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
#     rna = list(map(lambda x: encoder[x], rna))
#     rnas.append(rna)
# enc = OneHotEncoder()
# rnas = enc.fit_transform(rnas).toarray()
# X_train, X_test, y_train, y_test = train_test_split(rnas, label, test_size=0.2, random_state=42)
# trainset = list(zip(X_train, y_train))
# testset = list(zip(X_test, y_test))
# print("Load dataset finished!")
dic = {}
for line in open('../../../dataset/second_strcuture', 'r').readlines():
    rna, second = line.split('\t')
    dic[rna] = second[:-1]

data, label = get_data_sep(data_path="../../../dataset/trainset/")
trainset, testset, valset = [], [], []
for i in range(37):
    X_train, X_test, y_train, y_test = train_test_split(data[i], label[i], test_size=0.125, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 7, random_state=42)
    trainset.append(list(zip(X_train, y_train)))
    testset.append(list(zip(X_test, y_test)))
    valset.append(list(zip(X_val, y_val)))
print("Load dataset finished!")


class Simple_Deep:
    def __init__(self, path, para, trainset, testset):
        self.graph = tf.Graph()
        self.prediction, self.trainstep, self.loss = None, None, None
        self._path = path
        self._save_path, self._logs_path = None, None
        self.para = para
        self.trainset = trainset
        self.testset = testset
        self.predict_threshold = None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.local_initializer = tf.local_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/%s' % (self._path, str(time.time()))
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _initialize_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        self.sess.run(self.local_initializer)

    def _define_inputs(self):
        self.input = tf.placeholder(
            tf.float32,
            shape=[None, self.para['dim']]
        )
        self.input_s = tf.placeholder(
            tf.float32,
            shape=[None, self.para['sdim']]
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None]
        )
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.predict_threshold = tf.placeholder(tf.float32, shape=[], name='threshold')

    def _build_graph(self):
        batchsize = tf.shape(self.input)[0]
        x = tf.reshape(self.input, [batchsize, self.para['len'], 4])
        x_s = tf.reshape(self.input_s, [batchsize, self.para['len'], 6])
        conv_s = tf.layers.conv1d(x_s, 16, kernel_size=4, activation=tf.nn.relu)
        conv = tf.layers.conv1d(x, 16, kernel_size=4, activation=tf.nn.relu)
        out = tf.layers.max_pooling1d(conv, 3, strides=3)
        out = tf.nn.dropout(out, self.keep_prob)
        out_s = tf.layers.max_pooling1d(conv_s, 3, strides=3)
        out_s = tf.nn.dropout(out_s, self.keep_prob)
        out = tf.concat([out, out_s], -1)
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob,
                                                output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob,
                                                output_keep_prob=self.keep_prob)
        cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, attn_length=20)
        cell_bw = tf.contrib.rnn.AttentionCellWrapper(cell_bw, attn_length=20)
        output = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, out, dtype=tf.float32)[0], 2)
        len = int(output.shape[1]) - 1
        output = tf.slice(output, [0, len, 0], [-1, 1, -1])
        output = tf.reshape(output, [-1, 2 * self.para['hidden_size']])
        output = tf.layers.dense(output, self.para['label_dim'])
        output = tf.reshape(output, [batchsize])
        prediction = tf.nn.sigmoid(output)
        self.prediction = prediction
        loss = tf.losses.log_loss(self.labels, prediction)
        weights = tf.trainable_variables()
        l1_reg = tf.contrib.layers.l1_regularizer(scale=5e-6)
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_reg, weights)
        loss += regularization_penalty
        optimizer = tf.train.AdamOptimizer(self.para['lr'])
        self.trainstep = optimizer.minimize(loss)
        self.loss = loss

        # digit_prediction = tf.cast(tf.sign(tf.add(tf.sign(self.prediction - self.predict_threshold), 1)), tf.int64)
        # weights = tf.sign(tf.add(self.labels, 1))
        # self.labels = tf.cast(self.labels, tf.int64)
        # self.test_accuracy = tf.contrib.metrics.accuracy(labels=self.labels, predictions=digit_prediction,
        #                                               weights=weights)

    def test(self, batch_size):
        batch_per_epoch = int(len(self.testset) / batch_size)

        start_position = 0
        losses = []
        preds = []
        labels = []
        for b in range(batch_per_epoch):
            x, y = zip(*self.testset[start_position: start_position + batch_size])
            start_position += batch_size
            y = np.array(y)
            feed_dict = {
                self.input: x,
                self.labels: y,
                self.keep_prob: 1,
                self.predict_threshold: 0
            }
            fetch = [self.loss, self.prediction]
            loss, pred = self.sess.run(fetch, feed_dict)
            losses.append(loss)
            labels += y.tolist()
            preds += pred.tolist()
            # print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
        auc = roc_auc_score(labels, preds)
        print("Test loss {0}  auc {1}".format(np.mean(losses), auc))
        f = open(self.logs_path + '/log', 'a')
        f.write("Test loss {0} auc {1}\n".format(np.mean(losses), auc))

        return auc

    def train(self, batch_size, epoch):
        batch_per_epoch = int(len(self.trainset) / batch_size)
        max = 0
        for e in range(epoch):
            start_position = 0
            losses = []
            preds = []
            labels = []
            for b in range(batch_per_epoch):
                x, y = zip(*self.trainset[start_position: start_position + batch_size])
                start_position += batch_size
                y = np.array(y)
                feed_dict = {
                    self.input: x,
                    self.labels: y,
                    self.keep_prob: 0.5,
                    self.predict_threshold: 0
                }
                fetch = [self.trainstep, self.loss, self.prediction]
                _, loss, pred = self.sess.run(fetch, feed_dict)
                losses.append(loss)
                labels += y.tolist()
                preds += pred.tolist()
                # print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
            auc = roc_auc_score(labels, preds)
            print(
                "Train epoch {0} loss {1} auc {2}".format(e, np.mean(losses), auc))
            f = open(self.logs_path + '/log', 'a')
            f.write("Train epoch {0} loss {1} auc {2}\n".format(e, np.mean(losses), auc))
            result = self.test(int(sys.argv[3]))
            if result > max:
                model.save_model()
                max = result

    def get_aucs(self, batch_size):
        batch_per_epoch = int(len(self.testset) / batch_size)

        start_position = 0
        losses = []
        preds = []
        labels = []
        for b in range(batch_per_epoch):
            x, y = zip(*self.testset[start_position: start_position + batch_size])
            start_position += batch_size
            y = np.array(y)
            feed_dict = {
                self.input: x,
                self.labels: y,
                self.keep_prob: 1,
                self.predict_threshold: 0
            }
            fetch = [self.loss, self.prediction]
            loss, pred = self.sess.run(fetch, feed_dict)
            losses.append(loss)
            labels += y.tolist()
            preds += pred.tolist()
            # print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
        print(aucs(labels, preds))

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)


if __name__ == '__main__':
    protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3',
                    'EWSR1',
                    'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                    'LIN28B',
                    'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6',
                    'U2AF65',
                    'WTAP', 'ZC3H7B']
    para = {'len': 300, 'label_dim': 1, 'dim': 1200, 'hidden_size': 256, 'lr': float(sys.argv[4]), 'sdim': 1800}
    for i in range(37):
        path = './model_2structure/' + protein_list[i]
        if os.path.exists(path):
            continue
        os.mkdir(path)
        model = Simple_Deep(path, para, trainset[i], valset[i])
        model.train(batch_size=int(sys.argv[2]), epoch=int(sys.argv[1]))
        # model.load_model()
        # model.get_aucs(100)
        # model.test(100)
