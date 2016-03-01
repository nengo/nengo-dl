import nengo
import lasagne as lgn

from .simulator import Simulator
from .utils import default_config, settings, to_array

nonlinearity_map = {nengo.RectifiedLinear: lgn.nonlinearities.rectify,
                    nengo.Sigmoid: lgn.nonlinearities.sigmoid,
                    nengo.Direct: lgn.nonlinearities.linear}


