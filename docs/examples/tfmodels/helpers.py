# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """
    Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    id_to_word = {v: k for k, v in word_to_id.items()}
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)

    return {'train_data': train_data,
            'valid_data': valid_data,
            'test_data': test_data,
            'vocabulary': vocabulary,
            'word_to_id': word_to_id,
            'id_to_word': id_to_word}


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
    Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data)
    batch_len = len(raw_data) // batch_size

    data = np.reshape(raw_data[:batch_size * batch_len],
                      (batch_size, batch_len))

    epoch_size = (batch_len - 1) // num_steps
    assert epoch_size > 0

    for i in range(epoch_size):

        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]

        x.reshape((batch_size, num_steps))
        y.reshape((batch_size, num_steps))

        yield x, y
