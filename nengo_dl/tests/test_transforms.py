# pylint: disable=missing-docstring

from distutils.version import LooseVersion

import nengo
import numpy as np
import pytest


@pytest.mark.skipif(LooseVersion(nengo.__version__) <= "2.8.0",
                    reason="Nengo Convolutions not implemented")
@pytest.mark.parametrize("channels_last", (True, False))
def test_merge_conv(Simulator, channels_last, seed, pytestconfig):
    from nengo.builder.transforms import ConvInc

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(np.ones(32))
        b = nengo.Node(size_in=12)
        c = nengo.Node(size_in=12)
        nengo.Connection(a, b, synapse=None, transform=nengo.Convolution(
            3, (4, 4, 2) if channels_last else (2, 4, 4),
            channels_last=channels_last))
        nengo.Connection(a, c, synapse=None, transform=nengo.Convolution(
            3, (4, 4, 2) if channels_last else (2, 4, 4),
            channels_last=channels_last))
        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    with pytest.warns(None) as recwarns:
        with Simulator(net) as sim:
            assert len([ops for ops in sim.tensor_graph.plan
                        if isinstance(ops[0], ConvInc)]) == 1

            sim.step()

    # check for warning about force_last
    # note: this also assures us that we are testing on the GPU in native
    # channels_first when possible
    recwarns = [w for w in recwarns if "channels_last=False" in str(w.message)]
    if channels_last or (pytest.gpu_installed
                         and pytestconfig.getoption("--device") != "/cpu:0"):
        assert len(recwarns) == 0
    else:
        assert len(recwarns) == 1

    with nengo.Simulator(net) as canonical:
        canonical.step()

    assert np.allclose(sim.data[p_b], canonical.data[p_b])
    assert np.allclose(sim.data[p_c], canonical.data[p_c])


@pytest.mark.skipif(LooseVersion(nengo.__version__) <= "2.8.0",
                    reason="Nengo Convolutions not implemented")
@pytest.mark.parametrize("d", (3, 4))
def test_conv_error(Simulator, d):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, transform=nengo.Convolution(
            1, [1] * (d + 1), kernel_size=[1] * d, strides=[1] * d))

    try:
        with Simulator(net):
            pass
    except NotImplementedError:
        assert d == 4
    else:
        assert d == 3
