import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from code.feature_engineering import get_data

data, label = get_data(data_path="../../../dataset/RNA_trainset/")

rnas = []
for rna in data:
    encoder = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    rna = list(map(lambda x: encoder[x], rna))
    rnas.append(rna)
enc = OneHotEncoder()
rnas = enc.fit_transform(rnas).toarray()
X_train, X_test, y_train, y_test = train_test_split(rnas, label, test_size=0.2, random_state=42)
trainset = list(zip(X_train, y_train))
testset = list(zip(X_test, y_test))
print("Load dataset finished!")

class Simple_Deep:
    def __init__(self, path, para, trainset, testset):
        self.graph = tf.Graph()
        self.prediction, self.trainstep, self.loss = None, None, None
        self._path = path
        self.para = para
        self.trainset = trainset
        self.testset = testset
        self.predict_threshold = None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
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
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _initialize_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)

    def _define_inputs(self):
        self.input = tf.placeholder(
            tf.float32,
            shape=[None, self.para['dim']]
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
        batchsize = tf.shape(self.input)[0]
        x = tf.reshape(self.input, [batchsize, self.para['len'], 4])
        # x = tf.transpose(x, [0, 2, 1])
        # filter = tf.Variable(tf.random_normal([tf.shape(x)[1], 4, 1, 1]))
        conv = tf.layers.conv1d(x, 16, kernel_size=4, activation=tf.nn.relu)
        out = tf.layers.max_pooling1d(conv, 3, strides=3)
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.para['hidden_size'])
        output = tf.concat(tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, out, dtype=tf.float32)[0], 2)
        len = int(output.shape[1]) - 1
        output = tf.slice(output, [0, len, 0], [-1, 1, -1])
        output = tf.reshape(output, [-1, 2 * self.para['hidden_size']])
        output = tf.layers.dense(output, self.para['label_dim'])
        self.prediction = tf.nn.sigmoid(output)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=output)
        loss = tf.reduce_sum(tf.multiply(loss, self.mask)) / tf.reduce_sum(self.mask)
        optimizer = tf.train.AdamOptimizer(self.para['lr'])
        self.trainstep = optimizer.minimize(loss)
        self.loss = loss

        digit_prediction = tf.sign(self.prediction - self.predict_threshold)
        weights = tf.sign(self.labels + 1)
        self.test_accuracy = tf.metrics.accuracy(labels=self.labels, predictions=digit_prediction, weights=weights)


    def train(self, batch_size, epoch):
        batch_per_epoch = int(len(self.trainset) / batch_size)
        start_position = 0
        for e in range(epoch):
            for b in range(batch_per_epoch):
                x, y = zip(*trainset[start_position: start_position + batch_size])
                start_position += batch_size
                y = np.array(y)
                mask = y != -1
                mask = mask.astype(np.float32)
                feed_dict = {
                    self.input: x,
                    self.labels: y,
                    self.mask: mask,
                    self.keep_prob: 0.5
                }
                fetch = [self.trainstep, self.loss, self.prediction]
                _, loss, pred = self.sess.run(fetch, feed_dict)
                print("Train epoch {0} batch {1} loss {2}".format(e, b, loss))
            x, y = zip(*testset)
            feed_dict = {
                self.input: x,
                self.labels: y,
                self.keep_prob: 1
            }
            accuracy = self.sess.run(fetches=[self.test_accuracy], feed_dict=feed_dict)
            print("Test epoch {0} accuracy {1}".format(e, accuracy))


if __name__ == '__main__':
    para = {'len': 300, 'label_dim': 37, 'dim': 1200, 'hidden_size': 256, 'lr': 1e-4}
    model = Simple_Deep('./model', para, trainset, testset)
    model.train(batch_size=100, epoch=5)
