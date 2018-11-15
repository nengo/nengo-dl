# pylint: disable=missing-docstring

from nengo.exceptions import BuildError
import pytest

from nengo_dl.builder import Builder, OpBuilder, NengoModel
from nengo_dl.tests import dummies


def test_custom_builder():
    class TestOp:
        sets = None
        incs = None
        reads = None
        updates = None

    ops = (TestOp(),)

    # error if no builder registered
    with pytest.raises(BuildError):
        Builder.pre_build(ops, None, None, None)

    # error if no pre-built object
    with pytest.raises(BuildError):
        Builder.build(ops, None, {})

    # warning if builder doesn't subclass OpBuilder
    with pytest.warns(UserWarning):
        @Builder.register(TestOp)
        class TestOpBuilder0:  # pylint: disable=unused-variable
            pass

    # warning when overwriting a registered builder
    with pytest.warns(UserWarning):
        @Builder.register(TestOp)
        class TestOpBuilder(OpBuilder):
            pre_built = False
            post_built = False

            def __init__(self, ops, signals, config):
                super(TestOpBuilder, self).__init__(ops, signals, config)
                self.pre_built = True

            def build_step(self, signals):
                assert self.pre_built
                assert not self.post_built

                return 0, 1

            def build_post(self, ops, signals, sess, rng):
                self.post_built = True

    op_builds = {}
    Builder.pre_build(ops, None, op_builds, None)

    result = Builder.build(ops, None, op_builds)

    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 1

    op_builds[ops].build_post(None, None, None, None)
    assert op_builds[ops].post_built


@pytest.mark.parametrize("fail_fast", (True, False))
def test_custom_model(fail_fast):
    model = NengoModel(fail_fast=fail_fast)

    try:
        model.add_op(dummies.Op())
    except NotImplementedError:
        assert fail_fast
    else:
        assert not fail_fast
