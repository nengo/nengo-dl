
"""
This is a work-in-progress LSTM language model implemented with TensorFlow.

The Penn Treebank data required to use this model can be obtained as follows:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To subsequently run the model:
$ python ptb_lm.py simple-examples/data
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib.seq2seq import sequence_loss
from helpers import ptb_producer, ptb_raw_data


class PTBModel(object):
    """LSTM language model for use with the Penn Treebank dataset. This model
    is very loosely adapted from code accompanying the Tensorflow tutorial at
    https://www.tensorflow.org/tutorials/recurrent. The goal is to provide a
    minimal example of a language model in tensorflow that can be used to
    predict sentence continuations with simple multi-word prompts.

    Parameters
    ----------
    path : str
        The absolute path to a directory containing the PTB text final_states
    dim : int
        The dimensionality of the LSTM state vectors.
    """
    def __init__(self, path, dim):
        self.ptb_data = ptb_raw_data(path)
        self.word_to_id = self.ptb_data['word_to_id']
        self.id_to_word = self.ptb_data['id_to_word']
        self.vocab = self.ptb_data['vocabulary']
        self.dim = dim
        self.producer = ptb_producer

    def _start_session(self):
        if not hasattr(self, 'sess'):
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.initializer)

    def build_graph(self, max_grad=5):
        '''Build the computation graph to be executed during model training
        Parameters
        ----------
        max_grad : int (optional)
            The maximum gradient norm to use for gradient clipping
        '''
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.xs = tf.placeholder(tf.int32, shape=[None, None])
            self.ys = tf.placeholder(tf.int32, shape=[None, None])
            self.init = tf.placeholder(tf.float32, shape=[2, None, self.dim])

            b_size = tf.shape(self.xs)[0]
            n_steps = tf.shape(self.xs)[1]

            e_matrix = tf.Variable(tf.random_uniform([self.vocab, self.dim],
                                   -1.0, 1.0))

            embeddings = tf.nn.embedding_lookup(e_matrix, self.xs)

            cell = tf.contrib.rnn.BasicLSTMCell(self.dim, state_is_tuple=True)
            state = tf.nn.rnn_cell.LSTMStateTuple(self.init[0], self.init[1])

            outputs, self.last_state = tf.nn.dynamic_rnn(cell, embeddings,
                                                         initial_state=state,
                                                         dtype=tf.float32)

            output = tf.reshape(tf.concat(outputs, 1), [-1, self.dim])

            b_softmax = tf.Variable(tf.zeros(self.vocab))
            W_softmax = tf.Variable(tf.random_uniform([self.dim, self.vocab],
                                    -0.1, 0.1))

            logits = tf.nn.xw_plus_b(output, W_softmax, b_softmax)
            logits = tf.reshape(logits, [b_size, n_steps, self.vocab])
            self.probs = tf.nn.softmax(logits)

            ones = tf.ones([b_size, n_steps], dtype=tf.float32)
            loss = sequence_loss(logits, self.ys, ones,
                                 average_across_timesteps=False,
                                 average_across_batch=True)

            self.cost = tf.reduce_sum(loss)

            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad)

            optimizer = tf.train.GradientDescentOptimizer(1.0)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.initializer = tf.global_variables_initializer()

    def train(self, rate, epochs, b_size=20, n_steps=15, log_int=200):
        '''Trains the model for some number of epochs using the PTB dataset.

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
        log_int : int (optional)
            The batch interval for printing out info for perf monitoring
        '''
        self.total_cost = 0
        self.iterations = 0

        self._start_session()
        interval = log_int * n_steps
        zeros = np.zeros((2, b_size, self.dim))

        for epoch in range(epochs):
            data = self.producer(self.ptb_data['train_data'], b_size, n_steps)

            for words, labels in data:
                feed_dict = {self.xs: words, self.ys: labels, self.init: zeros}
                cost, _ = self.sess.run([self.cost, self.train_op], feed_dict)

                self.total_cost += cost
                self.iterations += n_steps

                if self.iterations % interval == 0:
                    perplexity = np.exp(self.total_cost / self.iterations)
                    print('Epoch: ', epoch)
                    print('Iters: ', self.iterations)
                    print('Train PPL: ', perplexity)
                    print('')

                    prompt = 'Financial markets rose in'
                    print(self.predict(prompt=prompt, n_steps=5))
                    print('')

    def predict(self, prompt, n_steps):
        '''Predict a continuation for some linguistic prompt

        Parameters
        ----------
        prompt : str
            A sequence of words for conditioning the LSTM's hidden state
        n_steps : int
            The number of time steps to run the LSTM to generate continuations
        '''
        words = [w.lower() for w in prompt.split()]
        assert all([word in self.word_to_id for word in words])

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

        return words, predictions

    def save(self, filename):
        '''Save the model to a checkpoint file'''
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, filename)

    def load(self, filename):
        '''Load an existing model from a checkpoint file'''
        with self.graph.as_default():
            self._start_session()
            saver = tf.train.Saver()
            saver.restore(self.sess, filename)


path = os.path.join(os.getcwd(), 'simple-examples/data')

b_size = 20
n_steps = 15
dim = 300
rate = 1.0

model = PTBModel(path, dim)
model.build_graph()
model.train(rate=rate, epochs=2, b_size=b_size, n_steps=n_steps)

model.save('ptb_model.ckpt')

print(model.predict('Trading stopped in the afternoon because', n_steps=6))
