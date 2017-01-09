import tensorflow as tf
from tensorflow.python.client.timeline import Timeline
import numpy as np

USE_WHILE = False
const_array = tf.constant(np.random.randn(100, 100))
const_idxs = tf.constant(np.random.permutation(100))
init_iter = tf.constant(0)
n = 1000

# unused_variable = tf.Variable(0)
# init_op = tf.global_variables_initializer()

def loop_cond(iter, *_):
    return iter < n


def loop_body(iter, val, idxs):
    iter += 1

    val = val[:]
    val = tf.gather(val, idxs)

    return iter, val, idxs


if USE_WHILE:
    loop_iter, loop_val, loop_idxs = tf.while_loop(
        loop_cond, loop_body, (init_iter, const_array, const_idxs))
else:
    loop_iter = init_iter
    loop_val = const_array
    loop_idxs = const_idxs
    for i in range(n):
        loop_iter, loop_val, loop_idxs = loop_body(loop_iter, loop_val,
                                                   loop_idxs)

with tf.Session() as sess:
    # sess.run(init_op)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    result = sess.run([loop_iter, loop_val, loop_idxs],
                      options=run_options, run_metadata=run_metadata)

    timeline = Timeline(run_metadata.step_stats)
    with open("tmp.json", "w") as f:
        f.write(timeline.generate_chrome_trace_format())

assert result[0] == n
# assert np.allclose(result[1], 10 * const_array)
# assert np.allclose(result[1][50:], 0)
