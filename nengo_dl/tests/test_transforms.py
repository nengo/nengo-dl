# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl.utils import tf_gpu_installed


@pytest.mark.parametrize("channels_last", (True, False))
def test_merge_conv(Simulator, channels_last, seed, pytestconfig):
    from nengo.builder.transforms import (  # pylint: disable=import-outside-toplevel
        ConvInc,
    )

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(np.ones(32))
        b = nengo.Node(size_in=12)
        c = nengo.Node(size_in=12)
        nengo.Connection(
            a,
            b,
            synapse=None,
            transform=nengo.Convolution(
                3,
                (4, 4, 2) if channels_last else (2, 4, 4),
                channels_last=channels_last,
            ),
        )
        nengo.Connection(
            a,
            c,
            synapse=None,
            transform=nengo.Convolution(
                3,
                (4, 4, 2) if channels_last else (2, 4, 4),
                channels_last=channels_last,
            ),
        )
        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    with pytest.warns(None) as recwarns:
        with Simulator(net) as sim:
            assert (
                len(
                    [
                        ops
                        for ops in sim.tensor_graph.plan
                        if isinstance(ops[0], ConvInc)
                    ]
                )
                == 1
            )

            sim.step()

    # check for warning about force_last
    # note: this also assures us that we are testing on the GPU in native
    # channels_first when possible
    recwarns = [w for w in recwarns if "channels_last=False" in str(w.message)]
    device = pytestconfig.getoption("--device")
    if channels_last or (
        tf_gpu_installed and (device is None or "gpu" in device.lower())
    ):
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
