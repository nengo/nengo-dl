# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf
from nengo.builder.transforms import ConvInc

from nengo_dl.compat import HAS_NENGO_3_2_0, ConvolutionTranspose, ConvTransposeInc
from nengo_dl.utils import tf_gpu_installed


@pytest.mark.parametrize("force_cpu", (False, True))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("transpose", (True, False))
def test_merge_conv(Simulator, transpose, channels_last, force_cpu, seed, pytestconfig):
    if transpose and not HAS_NENGO_3_2_0:
        pytest.skip("Nengo version does not support ConvolutionTranspose")

    device = pytestconfig.getoption("--device")
    is_cpu = not tf_gpu_installed or (device is not None and "cpu" in device.lower())
    if is_cpu and not force_cpu:
        pytest.skip("CPU will be tested in the `force_cpu` case")
    else:
        is_cpu = is_cpu or force_cpu

    def make_transform():
        if transpose:
            return ConvolutionTranspose(
                n_filters=7,
                input_shape=(4, 4, 5) if channels_last else (5, 4, 4),
                channels_last=channels_last,
            )
        return nengo.Convolution(
            n_filters=3,
            input_shape=(4, 4, 2) if channels_last else (2, 4, 4),
            channels_last=channels_last,
        )

    with nengo.Network(seed=seed) as net:
        transform1 = make_transform()
        transform2 = make_transform()
        a = nengo.Node(np.ones(transform1.input_shape.size))
        b = nengo.Node(size_in=transform1.output_shape.size)
        c = nengo.Node(size_in=transform2.output_shape.size)
        nengo.Connection(a, b, synapse=None, transform=transform1)
        nengo.Connection(a, c, synapse=None, transform=transform2)
        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    sim_kwargs = {"device": "/cpu:0"} if force_cpu else {}

    with pytest.warns(None) as recwarns:
        with Simulator(net, **sim_kwargs) as sim:
            conv_ops = [
                ops
                for ops in sim.tensor_graph.plan
                if isinstance(ops[0], (ConvInc, ConvTransposeInc))
            ]
            assert len(conv_ops) == 1

            sim.step()

    # check for warning about force_last
    # note: this also assures us that we are testing on the GPU in native
    # channels_first when possible
    recwarns = [w for w in recwarns if "channels_last=False" in str(w.message)]
    if channels_last or not is_cpu:
        assert len(recwarns) == 0
    else:
        assert len(recwarns) > 0

    with nengo.Simulator(net) as canonical:
        canonical.step()

    assert np.allclose(sim.data[p_b], canonical.data[p_b], atol=5e-6)
    assert np.allclose(sim.data[p_c], canonical.data[p_c], atol=5e-6)


@pytest.mark.parametrize("d", (3, 4))
def test_conv_error(Simulator, d):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Node(size_in=1)
        nengo.Connection(
            a,
            b,
            transform=nengo.Convolution(
                1, [1] * (d + 1), kernel_size=[1] * d, strides=[1] * d
            ),
        )

    try:
        with Simulator(net):
            pass
    except NotImplementedError:
        assert d == 4
    else:
        assert d == 3


@pytest.mark.training
def test_sparse(Simulator, rng):
    with nengo.Network() as net:
        # two parallel inputs so that we test the merging
        in0 = nengo.Node(rng.rand(3))
        in1 = nengo.Node(rng.rand(4))
        out_dense = nengo.Node(size_in=5)
        out_sparse = nengo.Node(size_in=5)
        p_dense = nengo.Probe(out_dense)
        p_sparse = nengo.Probe(out_sparse)

        w0 = rng.rand(5, 3)
        w1 = rng.rand(5, 4)

        # dense connections
        c_dense0 = nengo.Connection(in0, out_dense, transform=w0, synapse=None)
        c_dense1 = nengo.Connection(in1, out_dense, transform=w1, synapse=None)

        # sparse connections
        c_sparse0 = nengo.Connection(
            in0,
            out_sparse,
            transform=nengo.transforms.Sparse(
                indices=np.reshape(
                    np.dstack(np.meshgrid(np.arange(5), np.arange(3), indexing="ij")),
                    (-1, 2),
                ),
                init=w0.ravel(),
                shape=(5, 3),
            ),
            synapse=None,
        )
        c_sparse1 = nengo.Connection(
            in1,
            out_sparse,
            transform=nengo.transforms.Sparse(
                indices=np.reshape(
                    np.dstack(np.meshgrid(np.arange(5), np.arange(4), indexing="ij")),
                    (-1, 2),
                ),
                init=w1.ravel(),
                shape=(5, 4),
            ),
            synapse=None,
        )

    with Simulator(net) as sim:
        # check that operators are getting merged
        assert (
            len(
                [
                    p
                    for p in sim.tensor_graph.plan
                    if isinstance(p[0], nengo.builder.transforms.SparseDotInc)
                ]
            )
            == 1
        )

        # check that sparse and dense transforms produce the same result
        sim.run_steps(10)
        assert np.allclose(sim.data[p_dense], sim.data[p_sparse])

        # check that training on sparse and dense transforms produces the
        # same result
        sim.compile(tf.optimizers.SGD(0.01), loss=tf.losses.mse)
        sim.fit(
            {in0: np.ones((10, 5, 3)), in1: np.ones((10, 5, 4))},
            {p_dense: np.ones((10, 5, 5)), p_sparse: np.ones((10, 5, 5))},
            epochs=10,
        )
        assert np.allclose(
            sim.data[c_dense0].weights.ravel(), sim.data[c_sparse0].weights
        )
        assert np.allclose(
            sim.data[c_dense1].weights.ravel(), sim.data[c_sparse1].weights
        )
