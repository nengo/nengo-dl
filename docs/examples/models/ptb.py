
import os
import re

import numpy as np
import tensorflow as tf
import nengo.spa as spa

from tensorflow.contrib.seq2seq import sequence_loss
from .helpers import ptb_producer, ptb_raw_data


class LanguageModel(object):
    """LSTM language model for use with the Penn Treebank dataset. This model
    is very loosely adapted from code accompanying the Tensorflow tutorial at
    https://www.tensorflow.org/tutorials/recurrent. The goal is to provide a
    minimal example of a language model in tensorflow that can be used to
    predict sentence continuations with simple multi-word prompts.

    Parameters
    ----------
    path : str
        The absolute path to a directory containing the PTB text files
    dim : int
        The dimensionality of the LSTM state vectors.
    """
    def __init__(self, path, dim):
        self.ptb = ptb_raw_data(path)
        self.word_to_id = self.ptb['word_to_id']
        self.id_to_word = self.ptb['id_to_word']
        self.vsize = self.ptb['vocabulary']
        self.dim = dim
        self.build_graph()

    @staticmethod
    def has_punc(word):
        regex = re.compile(r"[0-9]|[^\w]")
        if regex.findall(word):
            return True

    def _start_session(self):
        """Creates a session object for executing a built computation graph"""
        if not hasattr(self, 'sess'):
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.initializer)

    def build_graph(self, max_grad=5):
        """Builds the computation graph to be executed during model training
        and inference.

        Parameters
        ----------
        max_grad : int (optional)
            The maximum gradient norm to use for gradient clipping
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.xs = tf.placeholder(tf.int32, shape=[None, None])
            self.ys = tf.placeholder(tf.int32, shape=[None, None])
            self.init = tf.placeholder(tf.float32, shape=[2, None, self.dim])

            b_size = tf.shape(self.xs)[0]
            n_steps = tf.shape(self.xs)[1]

            e_matrix = tf.Variable(tf.random_uniform([self.vsize, self.dim],
                                   -1.0, 1.0), name='embedding_matrix')

            embeddings = tf.nn.embedding_lookup(e_matrix, self.xs)

            cell = tf.contrib.rnn.BasicLSTMCell(self.dim, state_is_tuple=True)
            state = tf.nn.rnn_cell.LSTMStateTuple(self.init[0], self.init[1])

            outputs, self.last_state = tf.nn.dynamic_rnn(cell, embeddings,
                                                         initial_state=state,
                                                         dtype=tf.float32)

            output = tf.reshape(tf.concat(outputs, 1), [-1, self.dim])

            b_softmax = tf.Variable(tf.zeros(self.vsize), name='b_softmax')
            W_softmax = tf.Variable(tf.random_uniform([self.dim, self.vsize],
                                    -0.1, 0.1), name='W_softmax')

            logits = tf.nn.xw_plus_b(output, W_softmax, b_softmax)
            logits = tf.reshape(logits, [b_size, n_steps, self.vsize])
            self.probs = tf.nn.softmax(logits)

            ones = tf.ones([b_size, n_steps], dtype=tf.float32)
            loss = sequence_loss(logits, self.ys, ones,
                                 average_across_timesteps=False,
                                 average_across_batch=True)

            self.cost = tf.reduce_sum(loss)

            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad)

            optimizer = tf.train.GradientDescentOptimizer(1)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.initializer = tf.global_variables_initializer()

    def train(self, rate, epochs, b_size=20, n_steps=15):
        """Trains the model for some number of epochs using the PTB dataset.

        Parameters
        ----------
        rate : float
            The learning rate to use for weight updates
        epochs : int
            The number of complete passes through the PTB training data to do
        b_size : int (optional)
            The batch size to use during training
        n_steps : int (optional)
            The number of time steps to unroll the network during training,
            since we are doing truncated backpropogation through time.
        """
        self.total_cost = 0
        self.iterations = 0

        self._start_session()
        zeros = np.zeros((2, b_size, self.dim))

        for epoch in range(epochs):
            data = ptb_producer(self.ptb['train_data'], b_size, n_steps)

            for words, labels in data:
                feed_dict = {self.xs: words, self.ys: labels, self.init: zeros}
                cost, _ = self.sess.run([self.cost, self.train_op], feed_dict)

                self.total_cost += cost
                self.iterations += n_steps

            print('Epoch: ', epoch)
            print('Train PPL: ', np.exp(self.total_cost / self.iterations))
            print('')

    def predict(self, prompt, n_steps):
        """Predict a continuation for some linguistic prompt.

        Parameters
        ----------
        prompt : str
            A sequence of words for conditioning the LSTM's hidden state
        n_steps : int
            The number of time steps to run the LSTM to generate continuations
        """
        words = [w.lower() for w in prompt.split()]
        assert all([w in self.word_to_id for w in words])

        # for initializing the LSTM cell with a batch_size of 1
        state = np.zeros((2, 1, self.dim))

        for word in words:
            idx = self.word_to_id[word]
            idx = np.asarray(idx).reshape(1, 1)
            null = np.zeros((1, 1))

            feed_dict = {self.xs: idx, self.ys: null, self.init: state}

            probs, state = self.sess.run([self.probs, self.last_state],
                                         feed_dict)

            idx = np.argmax(probs[0, 0, :])

        predictions = []
        for step in range(n_steps):
            predictions.append(self.id_to_word[idx])

            idx = np.asarray(idx).reshape(1, 1)
            feed_dict = {self.xs: idx, self.ys: null, self.init: state}
            probs, state = self.sess.run([self.probs, self.last_state],
                                         feed_dict)

            idx = np.argmax(probs[0, 0, :])

        return predictions

    def perplexity_eval(self, data, b_size=20, n_steps=15):
        """Compute the average perplexity on a PTB-style dataset

        Parameters
        ----------
        data : list of int
            A list of word ids constituting a text file.
        b_size : int (optional)
            The batch size to use for computing perplexity
        n_steps : int (optional)
            The number of time steps to unroll the network
        """
        state = np.zeros((2, b_size, self.dim))

        total_cost = 0
        iterations = 0

        feed = ptb_producer(data, b_size, n_steps)

        for words, labels in feed:
            feed_dict = {self.xs: words, self.ys: labels, self.init: state}
            cost, state = self.sess.run([self.cost, self.last_state],
                                        feed_dict)
            total_cost += cost
            iterations += n_steps
            perplexity = np.exp(total_cost / iterations)

        return perplexity

    def save_variables(self, var_names, filename):
        """Save a list of named variables in the model to a checkpoint file,
        specifically for use in a Nengo DL tensornode. The naming logic here
        is a hack for this specific use case. TODO: Fix.

        Parameters
        ----------
        var_names : list of str
            A list of variable names included in self.graph (errors otherwise)
        b_size : str
            The name of a checkpoint file to save the variables to
        """
        with self.graph.as_default():
            tn_scope = 'SimTensorNodeBuilder/'
            all_vars = tf.global_variables()

            var_list = {}

            for n in var_names:
                if 'rnn' in n:
                    var_list[n] = [v for v in all_vars if n in v.name].pop()
                else:
                    var_list[tn_scope + n] = [v for v in all_vars
                                              if n in v.name].pop()

            saver = tf.train.Saver(var_list=var_list)
            saver.save(self.sess, filename)

    def save(self, filename):
        """Save the model to a checkpoint file"""
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, filename)

    def load(self, filename):
        """Load an existing model from a checkpoint file"""
        with self.graph.as_default():
            self._start_session()
            saver = tf.train.Saver()
            saver.restore(self.sess, filename)

    def build_spa_vocab(self, dim):
        """Build a spa Vocabulary using the model vocabulary"""
        vocab = spa.Vocabulary(dim, max_similarity=1.0)

        for word, idx in self.word_to_id.items():
            if not self.has_punc(word):
                vocab.parse(word.upper())

        return vocab


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'simple-examples/data')

    b_size = 20
    n_steps = 15
    dim = 300
    rate = 1.0

    model = LanguageModel(path, dim)
    model.train(rate=rate, epochs=5, b_size=b_size, n_steps=n_steps)

    model.save('ptb_model.ckpt')
