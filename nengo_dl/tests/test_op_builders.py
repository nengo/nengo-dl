# pylint: disable=missing-docstring

from nengo import builder
from nengo.builder import signal, operator


def test_elementwise_inc(Simulator):
    # note: normally the op_builders are just tested as part of the nengo
    # tests.  but in this particular case, there are no nengo tests that
    # have a scalar, non-1 transform.  those all get optimized out during
    # the graph optimization, so we don't end up with any tests of
    # elementwiseinc where A is a scalar. so that's what this is for.

    model = builder.Model()

    a = signal.Signal([2.0])
    x = signal.Signal([[3.0]])
    y = signal.Signal([[1.0]])
    op = operator.ElementwiseInc(a, x, y)
    model.add_op(op)

    with Simulator(None, model=model) as sim:
        sim.sess.run(sim.tensor_graph.steps_run,
                     feed_dict={sim.tensor_graph.step_var: 0,
                                sim.tensor_graph.stop_var: 5})
