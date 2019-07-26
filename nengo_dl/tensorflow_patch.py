"""
Applies monkey-patches to TensorFlow to fix bugs or add functionality.
"""

import copy
import traceback

from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import math_ops, array_ops, data_flow_ops

saved_registry = copy.copy(ops._gradient_registry._registry)


def patch_dynamic_stitch_grad():
    """Tensorflow's current gradient implementation for `tf.dynamic_stitch` is
    incorrect.  This monkey-patches TensorFlow to fix the bug."""

    def DynamicStitchGrads(op, grad):
        num_values = len(op.inputs) // 2
        indices_grad = [None] * num_values

        def AsInt32(x):
            return (
                x
                if op.inputs[0].dtype == dtypes.int32
                else math_ops.cast(x, dtypes.int32)
            )

        idxs = [
            AsInt32(array_ops.reshape(op.inputs[i], (-1,))) for i in range(num_values)
        ]
        if isinstance(grad, ops.IndexedSlices):
            output_shape = array_ops.shape(op.outputs[0])
            output_rows = output_shape[0]
            grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)

        values_grad = []
        zeros = array_ops.zeros_like(grad)
        idx_zeros = [zeros[: array_ops.shape(x)[0]] for x in idxs]
        grad_range = math_ops.range(array_ops.shape(grad)[0])
        for i in range(num_values):
            if i == num_values - 1:
                v_grad = grad
            else:
                v_grad = data_flow_ops.dynamic_stitch(
                    [grad_range] + idxs[i + 1 :], [grad] + idx_zeros[i + 1 :]
                )
            v_grad = array_ops.gather(v_grad, AsInt32(op.inputs[i]))
            values_grad += [v_grad]

        return indices_grad + values_grad

    # need to stick in the registry manually, to override the already
    # registered implementation
    ops._gradient_registry._registry["DynamicStitch"] = {
        "type": DynamicStitchGrads,
        "location": traceback.extract_stack(),
    }


def undo_patch():
    """Restores TensorFlow to its original state."""

    ops._gradient_registry._registry = copy.copy(saved_registry)
