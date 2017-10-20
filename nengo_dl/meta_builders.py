import copy
from collections import deque
import logging
import warnings

from nengo import builder, Connection
from nengo.builder.operator import ElementwiseInc, DotInc, Reset
from nengo.builder.processes import SimProcess
from nengo.exceptions import SimulationError
from nengo.synapses import LinearFilter
import tensorflow as tf

from nengo_dl import op_builders, process_builders
from nengo_dl.builder import Builder, OpBuilder

logger = logging.getLogger(__name__)


class CaptureOps(object):
    def __init__(self, model):
        self.captured_ops = []
        self.model = model

    def __getattr__(self, attr):
        if attr in ("model", "captured_ops", "add_op", "build"):
            return super(CaptureOps, self).__getattr__(attr)
        else:
            return getattr(self.model, attr)

    def add_op(self, op):
        self.captured_ops.append(op)

    def build(self, obj, *args, **kwargs):
        return self.builder.build(self, obj, *args, **kwargs)


def build_connection(model, conn):
    if (not getattr(model, "_in_nengo_dl", False) or
            conn.learning_rule is not None):
        # TODO: make learning rules work with SimConnection
        # for complicated connections we just fall back on the normal builder
        builder.connection.build_connection(model, conn)
    else:
        # build the connection into our dummy model that captures all the
        # operators
        capture_model = CaptureOps(model)
        builder.connection.build_connection(capture_model, conn)

        # we only build SimConnection up to the second last op, since that is
        # where the connection is probeable and where the synapse update
        # occurs. so add the last op back into the regular model.
        model.add_op(capture_model.captured_ops.pop())

        op = SimConnection(conn, capture_model.captured_ops,
                           model.time)
        model.add_op(op)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    builder.Builder.register(Connection)(build_connection)


class SimConnection(builder.Operator):
    """Operator that combines all the computations of a connection into a
    single op.
    """

    def __init__(self, conn, ops, time, tag=None):
        super(SimConnection, self).__init__(tag=str(conn))

        # we drop the reset ops, since they are never used
        self.ops = [o for o in ops if not isinstance(o, Reset)]
        self.conn = conn

        # collect external inputs/outputs from all ops in chain
        remaining_ops = deque(o for o in self.ops)

        # op is the first simpyfunc/dotinc/copy
        op = remaining_ops.popleft()
        reads = [o for o in op.reads]

        # add in the weights for the dotinc (if they weren't added above)
        if not isinstance(op, (ElementwiseInc, DotInc)):
            op = remaining_ops.popleft()
            reads.append(op.A)

        try:
            op = remaining_ops.popleft()

            # if there is an op left it should be a synapse
            assert isinstance(op, SimProcess)

            # some processes may need to read the sim time
            if not isinstance(op.process, LinearFilter):
                reads.append(time)

            outputs = op.updates
            is_update = True
        except IndexError:
            outputs = op.incs
            is_update = False

        self.sets = [] if is_update else outputs
        self.incs = []
        self.reads = reads
        self.updates = outputs if is_update else []

    def make_step(self, *args, **kwargs):
        """``make_step`` is never called by the NengoDL simulator, so if this
        is called it means that someone is trying to execute this op in
        some other Simulator."""

        def error():
            raise SimulationError("SimConnection can only be simulated in the "
                                  "NengoDL simulator")

        return error


@Builder.register(SimConnection)
class SimConnectionBuilder(OpBuilder):
    def __init__(self, ops, signals):
        super(SimConnectionBuilder, self).__init__(ops, signals)

        logger.debug([str(o) for o in zip(*[op.ops for op in ops])])

        self.prebuilt_ops = deque(
            Builder.builders[type(o[0])](o, signals)
            for o in zip(*[op.ops for op in ops]))

    def build_step(self, signals):
        remaining_ops = copy.copy(self.prebuilt_ops)
        simpyfunc_out = None

        logger.debug("building simconnection")
        logger.debug("\n".join(str(o) for o in remaining_ops))

        op_builder = remaining_ops.popleft()

        if isinstance(op_builder, op_builders.SimPyFuncBuilder):
            # apply simpyfunc (if pre is a direct ensemble)
            x = signals.gather(op_builder.input_data)
            x = op_builder._step([], x)
            simpyfunc_out = x

            op_builder = remaining_ops.popleft()
        elif isinstance(op_builder, op_builders.CopyBuilder):
            # apply copy slice (if pre_slice is an advanced index)
            x = signals.gather(op_builder.src_data)

            op_builder = remaining_ops.popleft()
        else:
            # gather input from the first weight dotinc op
            # TODO: why is this force copy necessary?
            x = signals.gather(op_builder.X_data, force_copy=True)

        # multiply by connection weights
        logger.debug("applying weights")
        assert isinstance(op_builder, (op_builders.ElementwiseIncBuilder,
                                       op_builders.DotIncBuilder))
        X_shape = op_builder.X_data.shape + (signals.minibatch_size,)
        Y_shape = op_builder.Y_data.shape + (signals.minibatch_size,)
        if x.get_shape() != X_shape:
            x = tf.reshape(x, X_shape)
        x = op_builder._step(signals.gather(op_builder.A_data), x)
        if x.get_shape() != Y_shape:
            x = tf.reshape(x, Y_shape)

        try:
            # apply synapse if there is one for this connection
            op_builder = remaining_ops.popleft()

            assert isinstance(op_builder, process_builders.SimProcessBuilder)

            logger.debug("applying synapse")

            if isinstance(op_builder.built_process,
                          process_builders.GenericProcessBuilder):
                x = op_builder.built_process._step(signals.time, x)
            else:
                x = op_builder.built_process._step(x, signals)

            signals.scatter(op_builder.built_process.output_data, x)
        except IndexError:
            signals.scatter(op_builder.Y_data, x)

        assert len(remaining_ops) == 0

        return simpyfunc_out

    def build_post(self, ops, *args):
        for i, op_builder in enumerate(self.prebuilt_ops):
            op_builder.build_post(tuple(o.ops[i] for o in ops), *args)
