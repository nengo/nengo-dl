import copy
import traceback

from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import (math_ops, array_ops, data_flow_ops,
                                   state_ops, variable_scope)

saved_registry = copy.copy(ops._gradient_registry._registry)


def patch_dynamic_stitch_grad():
    """Tensorflow's current gradient implementation for `tf.dynamic_stitch` is
    incorrect.  This monkey-patches Tensorflow to fix the bug."""

    def DynamicStitchGrads(op, grad):
        num_values = len(op.inputs) // 2
        indices_grad = [None] * num_values

        def AsInt32(x):
            return (x if op.inputs[0].dtype == dtypes.int32 else
                    math_ops.cast(x, dtypes.int32))

        idxs = [AsInt32(array_ops.reshape(op.inputs[i], (-1,)))
                for i in range(num_values)]
        if isinstance(grad, ops.IndexedSlices):
            output_shape = array_ops.shape(op.outputs[0])
            output_rows = output_shape[0]
            grad = math_ops.unsorted_segment_sum(grad.values, grad.indices,
                                                 output_rows)

        values_grad = []
        zeros = array_ops.zeros_like(grad)
        idx_zeros = [zeros[:array_ops.shape(x)[0]] for x in idxs]
        grad_range = math_ops.range(array_ops.shape(grad)[0])
        for i in range(num_values):
            if i == num_values - 1:
                v_grad = grad
            else:
                v_grad = data_flow_ops.dynamic_stitch(
                    [grad_range] + idxs[i + 1:], [grad] + idx_zeros[i + 1:])
            v_grad = array_ops.gather(v_grad, AsInt32(op.inputs[i]))
            values_grad += [v_grad]

        return indices_grad + values_grad

    # need to stick in the registry manually, to override the already
    # registered implementation
    ops._gradient_registry._registry["DynamicStitch"] = {
        "type": DynamicStitchGrads, "location": traceback.extract_stack()}


def patch_state_grads():
    """TensorFlow doesn't have a gradient implementation for state ops (e.g.,
    scatter_add/update).  This adds them in."""

    def ScatterUpdateGrads(op, grad):
        var, indices, updates = op.inputs

        updates_grad = array_ops.gather(grad, indices)

        # TODO: the dynamic_stitch approach might be faster if there were
        # a GPU dynamic_stitch implementation. should be available in tf 1.4
        # grad_range = math_ops.range(grad.get_shape()[0].value)
        # var_grad = data_flow_ops.dynamic_stitch(
        #     [grad_range, indices],
        #     [grad, array_ops.zeros(updates.get_shape())])

        if isinstance(grad, ops.IndexedSlices):
            # note: we could use this approach for everything, but the
            # temporary variable approach seems to be slightly faster (but we
            # can't use that on indexedslices)
            var_grad = grad - array_ops.scatter_nd(
                array_ops.expand_dims(indices, 1), updates_grad,
                var.get_shape())
        else:
            key = (tuple(grad.get_shape().as_list()), grad.dtype.base_dtype)

            with variable_scope.variable_scope(
                    "gradient_vars", reuse=variable_scope.AUTO_REUSE):
                var_grad = variable_scope.get_variable(
                    "tmp" + "_%s" * (len(grad.get_shape()) + 1) % (
                        key[0] + (key[1].name,)),
                    shape=key[0], dtype=key[1], trainable=False,
                    collections=["gradient_vars"])

            var_grad = state_ops.assign(var_grad, grad)
            var_grad = state_ops.scatter_update(
                var_grad, indices, array_ops.zeros_like(updates))
            var_grad = var_grad + 0  # TODO: why is this necessary?

        return var_grad, None, updates_grad

    def ScatterAddGrads(op, grad):
        _, indices, _ = op.inputs

        updates_grad = array_ops.gather(grad, indices)

        return grad, None, updates_grad

    def AssignGrads(_, grad):
        return array_ops.zeros_like(grad), grad

    def AssignAddGrads(_, grad):
        return grad, grad

    ops._gradient_registry._registry["ScatterUpdate"] = {
        "type": ScatterUpdateGrads, "location": traceback.extract_stack()}
    ops._gradient_registry._registry["ScatterAdd"] = {
        "type": ScatterAddGrads, "location": traceback.extract_stack()}
    ops._gradient_registry._registry["Assign"] = {
        "type": AssignGrads, "location": traceback.extract_stack()}
    ops._gradient_registry._registry["AssignAdd"] = {
        "type": AssignAddGrads, "location": traceback.extract_stack()}


def undo_patch():
    """Restores TensorFlow to its original state."""

    ops._gradient_registry._registry = copy.copy(saved_registry)
