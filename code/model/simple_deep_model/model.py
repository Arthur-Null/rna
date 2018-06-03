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


class Simple_Deep:
    def __init__(self, path, para, trainset, testset):
        self.graph = tf.Graph()

        self._path = path
        self.para = para
        self.trainset = trainset
        self.testset = testset
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
            tf.int32,
            shape=[None, self.para['label_dim']]
        )
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _build_graph(self):
        batchsize = tf.shape(self.input)[0]
        x = tf.reshape(self.input, [batchsize, self.para['len'], -1, 1])
        x = tf.transpose(x, [0, 2, 1, 3])
        # filter = tf.Variable(tf.random_normal([tf.shape(x)[1], 4, 1, 1]))
        conv = tf.layers.conv2d(x, 5, kernel_size=(4, 4), activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(conv, pool_size=(1, 4), strides=4)
        self.conv = out

    def test(self):
        x, y = zip(*trainset[0:6])
        feed_dict = {
            self.input: x,
            self.labels: y,
            self.mask: y,
            self.keep_prob: 0.5
        }
        fetch = self.conv
        s = self.sess.run(fetch, feed_dict)
        s = np.array(s)
        print(s.shape)


if __name__ == '__main__':
    para = {'len': 300, 'label_dim': 37, 'dim': 1200}
    model = Simple_Deep('./model', para, trainset, testset)
    model.test()