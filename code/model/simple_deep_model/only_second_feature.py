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

data, label = get_data(data_path="../../../dataset/RNA_trainset/")
dic = {}
for line in open('../../../dataset/second_strcuture', 'r').readlines():
    rna, second = line.split('\t')
    dic[rna] = second[:-1]

rnas = []
seconds = []
for rna in data:
    encoder = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    encoder_sec = {'S': 0, 'M': 1, 'H': 2, 'I': 3, 'T': 4, 'F': 5}
    second = dic[rna]
    second = list(map(lambda x: encoder_sec[x], second))
    rna = list(map(lambda x: encoder[x], rna))
    rnas.append(rna)
    seconds.append(second)
enc = OneHotEncoder(4)
rnas = enc.fit_transform(rnas).toarray()
enc2 = OneHotEncoder(6)
seconds = enc2.fit_transform(seconds).toarray()
data = list(seconds)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
trainset = list(zip(X_train, y_train))
testset = list(zip(X_test, y_test))
print("Load dataset finished!")


def cal_accuracy(label, pred, thethold=0.5):
    total = 0.
    match = 0.
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] != -1:
                total += 1
                if label[i][j] == 0 and pred[i][j] < thethold or label[i][j] == 1 and pred[i][j] >= thethold:
                    match += 1
    return match / total


def ave_auc(label, pred):
    auc = []
    l = [[]] * 37
    p = [[]] * 37
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] != -1:
                l[j].append(label[i][j])
                p[j].append(pred[i][j])
    for i in range(37):
        auc.append(roc_auc_score(l[i], p[i]))
    return np.mean(auc)


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
        self.input_s = tf.placeholder(
            tf.float32,
            shape=[None, self.para['sdim']]
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.para['label_dim']]
        )
        self.mask = tf.placeholder(
            tf.float32,
            shape=[None, self.para['label_dim']]
        )
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.predict_threshold = tf.placeholder(tf.float32, shape=[], name='threshold')

    def _build_graph(self):
        batchsize = tf.shape(self.input_s)[0]

        x_s = tf.reshape(self.input_s, [batchsize, self.para['len'], 6])
        conv_s = tf.layers.conv1d(x_s, 16, kernel_size=4, activation=tf.nn.relu)

        out_s = tf.layers.max_pooling1d(conv_s, 3, strides=3)

        out_s = tf.nn.dropout(out_s, self.keep_prob)

        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob,
                                                output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob,
                                                output_keep_prob=self.keep_prob)
        cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, attn_length=20)
        cell_bw = tf.contrib.rnn.AttentionCellWrapper(cell_bw, attn_length=20)
        output = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, out_s, dtype=tf.float32)[0], 2)
        len = int(output.shape[1]) - 1
        output = tf.slice(output, [0, len, 0], [-1, 1, -1])
        output = tf.reshape(output, [-1, 2 * self.para['hidden_size']])
        output = tf.layers.dense(output, self.para['label_dim'])
        self.prediction = tf.nn.sigmoid(output)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=output)
        loss = tf.reduce_sum(tf.multiply(loss, self.mask)) / tf.reduce_sum(self.mask)
        weights = tf.trainable_variables()
        l1_reg = tf.contrib.layers.l1_regularizer(scale=1e-6)
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
            x, y = zip(*testset[start_position: start_position + batch_size])
            start_position += batch_size

            y = np.array(y)
            mask = y != -1
            mask = mask.astype(np.float32)
            feed_dict = {
                self.input_s: x,
                self.labels: y,
                self.mask: mask,
                self.keep_prob: 1,
                self.predict_threshold: 0
            }
            fetch = [self.loss, self.prediction]
            loss, pred = self.sess.run(fetch, feed_dict)
            losses.append(loss)
            labels += y.tolist()
            preds += pred.tolist()
            # print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
        print("Test loss {0} accuracy {1} auc {2}".format(np.mean(losses), cal_accuracy(labels, preds),
                                                          ave_auc(labels, preds)))
        f = open(self.logs_path + '/log', 'a')
        f.write("Test loss {0} accuracy {1} auc {2}\n".format(np.mean(losses), cal_accuracy(labels, preds),
                                                              ave_auc(labels, preds)))

        return ave_auc(labels, preds)

    def train(self, batch_size, epoch):
        batch_per_epoch = int(len(self.trainset) / batch_size)
        max = 0
        for e in range(epoch):
            start_position = 0
            losses = []
            preds = []
            labels = []
            for b in range(batch_per_epoch):
                x, y = zip(*trainset[start_position: start_position + batch_size])
                start_position += batch_size

                y = np.array(y)
                mask = y != -1
                mask = mask.astype(np.float32)
                feed_dict = {
                    self.input_s: x,
                    self.labels: y,
                    self.mask: mask,
                    self.keep_prob: 0.5,
                    self.predict_threshold: 0
                }
                fetch = [self.trainstep, self.loss, self.prediction]
                _, loss, pred = self.sess.run(fetch, feed_dict)
                losses.append(loss)
                labels += y.tolist()
                preds += pred.tolist()
                # print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
            print(
                "Train epoch {0} loss {1} accuracy {2} auc {3}".format(e, np.mean(losses), cal_accuracy(labels, preds),
                                                                       ave_auc(labels, preds)))
            f = open(self.logs_path + '/log', 'a')
            f.write("Train epoch {0} loss {1} accuracy {2} auc {3}\n".format(e, np.mean(losses),
                                                                             cal_accuracy(labels, preds),
                                                                             ave_auc(labels, preds)))
            result = self.test(int(sys.argv[3]))
            if result > max:
                model.save_model()
                max = result

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)


if __name__ == '__main__':
    para = {'len': 300, 'label_dim': 37, 'dim': 1200, 'hidden_size': 512, 'lr': float(sys.argv[4]), 'sdim': 1800}
    model = Simple_Deep('./model_second', para, trainset, testset)
    model.train(batch_size=int(sys.argv[2]), epoch=int(sys.argv[1]))
