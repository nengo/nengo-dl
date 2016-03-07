import nengo
import lasagne as lgn

from . import builder, dists, layers, simulator
from .layers import LasagneNode
from .simulator import Simulator
from .utils import default_config, to_array

init = dists.InitWrapper()
nl_map = {nengo.RectifiedLinear: lgn.nonlinearities.rectify,
          nengo.Sigmoid: lgn.nonlinearities.sigmoid,
          nengo.Direct: lgn.nonlinearities.linear}
