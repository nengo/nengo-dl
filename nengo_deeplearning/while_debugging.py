import tensorflow as tf
from tensorflow.python.client.timeline import Timeline
import numpy as np

# USE_WHILE = False
# const_array = tf.constant(np.random.randn(100, 100))
# const_idxs = tf.constant(np.random.permutation(100))
# init_iter = tf.constant(0)
# n = 1000
#
#
# def loop_cond(iter, *_):
#     return iter < n
#
#
# def loop_body(iter, val, idxs):
#     iter += 1
#
#     val = val[:]
#     val = tf.gather(val, idxs)
#
#     return iter, val, idxs
#
#
# if USE_WHILE:
#     loop_iter, loop_val, loop_idxs = tf.while_loop(
#         loop_cond, loop_body, (init_iter, const_array, const_idxs))
# else:
#     loop_iter = init_iter
#     loop_val = const_array
#     loop_idxs = const_idxs
#     for i in range(n):
#         loop_iter, loop_val, loop_idxs = loop_body(loop_iter, loop_val,
#                                                    loop_idxs)
#
# with tf.Session() as sess:
#     # sess.run(init_op)
#
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     result = sess.run([loop_iter, loop_val, loop_idxs],
#                       options=run_options, run_metadata=run_metadata)
#
#     timeline = Timeline(run_metadata.step_stats)
#     with open("tmp.json", "w") as f:
#         f.write(timeline.generate_chrome_trace_format())
#
# assert result[0] == n
# assert np.allclose(result[1], 10 * const_array)
# assert np.allclose(result[1][50:], 0)

n = 1000
USE_WHILE = True

start = tf.constant([10])
stop = tf.constant([20])
stride = tf.constant([1])
idxs = tf.range(10, 20)
ones = tf.constant(np.ones(10), dtype=tf.float32)
with tf.variable_scope("my_variables"):
    var = tf.get_variable(
        "var", initializer=tf.constant_initializer(np.ones(30)), shape=(30,))
    var2 = tf.get_variable(
        "var2", initializer=tf.constant_initializer(np.zeros(30)), shape=(30,))


def loop_cond(step, *_):
    return step < n


def loop_body(step, var, var2):
    a = tf.strided_slice(var, start, stop, stride)
    var2 = tf.scatter_add(var2, idxs, a)
    with tf.control_dependencies([a]):
        var = tf.scatter_add(var, idxs, ones)

    with tf.control_dependencies([var, var2]):
        step += 1

    return step, var, var2


loop_i = tf.constant(0)
if USE_WHILE:
    loop_i, var, var2 = tf.while_loop(
        loop_cond, loop_body, [loop_i, var._ref(), var2._ref()])
else:
    for i in range(n):
        loop_i = loop_body(loop_i)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    end_step, end_var, end_var2 = sess.run([loop_i, var, var2])

    assert end_step == n
    print(end_var)
    print(end_var2)
    print(n * (n + 1) / 2)
    assert np.all(end_var2[10:20] == n * (n + 1) / 2)
