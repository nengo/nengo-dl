import lasagne as lgn
from nengo.dists import Distribution


class InitWrapper:
    def __getattr__(self, key):
        return wrap_lasagne(getattr(lgn.init, key))


def wrap_lasagne(dist):
    class WrappedDist(Distribution):
        def __init__(self, *args, **kwargs):
            self.lgn_dist = dist(*args, **kwargs)

        def sample(self, n, d=None, rng=None):
            if rng is not None:
                old_rng = lgn.random.get_rng()
                lgn.random.set_rng(rng)

            result = self.lgn_dist.sample(self._sample_shape(n, d))

            if rng is not None:
                lgn.random.set_rng(old_rng)

            return result

    WrappedDist.__name__ = "Wrapped_%s" % dist

    return WrappedDist


def wrap_nengo(dist):
    class WrappedDist(lgn.init.Initializer):
        def __init__(self, *args, **kwargs):
            self.nengo_dist = dist(*args, **kwargs)

        def __call__(self, shape):
            return self.sample(shape)

        def sample(self, shape):
            assert len(shape) == 2

            return self.nengo_dist.sample(shape[0], shape[1],
                                          rng=lgn.random.get_rng())

    WrappedDist.__name__ = "Wrapped_%s" % dist

    return WrappedDist
