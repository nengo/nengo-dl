TensorNodes
============

TensorNodes allow you to define parts of your model using TensorFlow, and then
insert those elements into a Nengo model.  TensorNodes work very similarly to a
regular :class:`~nengo:nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

Here is a simple example that uses a TensorNode to compute a ``sin`` wave:

.. code-block:: python

    import nengo
    import nengo_deeplearning as nengo_dl
    import tensorflow as tf

    with nengo.Network() as net:
        node = nengo_dl.TensorNode(lambda t: tf.sin(t))
        p = nengo.Probe(node)

    with nengo_dl.Simulator(net) as sim:
        sim.run_steps(1.0)

Note that probing and connecting to TensorNodes works in the same way as
regular Nodes.

However, computing ``sin`` is something we could do with a regular Node.  A
more useful application of TensorNodes is in defining network structures that
are not easily expressed in Nengo, such as convolutional neural networks.
For example, here is a network that applies a convolutional
layer to MNIST images:

.. code-block:: python

  import nengo
  import nengo_deeplearning as nengo_dl
  import numpy as np
  import tensorflow as tf
  from tensorflow.examples.tutorials.mnist import input_data

  class ConvLayer(object):
      def pre_build(self, shape_in, shape_out):
          self.n_mini = shape_in[0] # minibatch size
          self.img_size = int(np.sqrt(shape_in[1])) # image height/width
          self.n_channels = 32 # number of convolutional filters
          self.kernel_size = 3 # convolutional filter size
          self.size_out = shape_out[1] # output dimensionality

      def __call__(self, t, x):
          # reshape input signal to image shape
          image = tf.reshape(x, (self.n_mini, self.img_size, self.img_size, 1))

          # apply convolutional layer
          conv = tf.contrib.layers.conv2d(image, self.n_channels, self.kernel_size)

          # apply dense layer
          dense = tf.contrib.layers.flatten(conv)
          dense = tf.contrib.layers.fully_connected(dense, self.size_out)

          return dense

  with nengo.Network() as net:
      # load input data (mnist images)
      mnist = input_data.read_data_sets("MNIST_data/")

      # create node to feed in images
      inp = nengo.Node(nengo.processes.PresentInput(mnist.train.images, 1))

      # create TensorNode to insert the network defined in `ConvLayer`
      tf_node = nengo_dl.TensorNode(ConvLayer(), size_in=28 * 28, size_out=10)

      # create connections to/from TensorNodes, or probe their output, just
      # like a regular Node
      nengo.Connection(inp, tf_node)
      p = nengo.Probe(tf_node)

Note that the above example takes advantage of the ``pre_build`` feature of
TensorNodes.  If the object passed to TensorNode has a ``pre_build`` function,
NengoDL will call that function once when the model is constructed, and it will
pass in the shape of the input and output Tensors.  This can be used to
define any constants or other operations that don't need to be executed every
simulation timestep.

.. autoclass:: nengo_deeplearning.tensor_node.TensorNode