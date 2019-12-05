# pylint: disable=missing-docstring

from nengo.exceptions import BuildError
import pytest

from nengo_dl.builder import Builder, OpBuilder, NengoModel
from nengo_dl.tests import dummies
from nengo_dl.utils import NullProgressBar


def test_custom_builder():
    class TestOp:
        sets = None
        incs = None
        reads = None
        updates = None

    ops = (TestOp(),)
    builder = Builder([ops], None, None)
    progress = NullProgressBar()

    # error if no builder registered
    with pytest.raises(BuildError):
        builder.build_pre()

    # warning if builder doesn't subclass OpBuilder
    with pytest.warns(UserWarning):

        @Builder.register(TestOp)
        class TestOpBuilder0:  # pylint: disable=unused-variable
            pass

    # warning when overwriting a registered builder
    with pytest.warns(UserWarning):

        @Builder.register(TestOp)  # pylint: disable=unused-variable
        class TestOpBuilder(OpBuilder):
            pre_built = False
            post_built = False

            def __init__(self, ops, signals, config):
                super().__init__(ops, signals, config)
                self.pre_built = True

            def build_step(self, signals):
                assert self.pre_built
                assert not self.post_built

                return 0, 1

            def build_post(self, ops, signals, config):
                self.post_built = True

    builder.build_pre(progress)

    result = builder.build(progress)

    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 1

    builder.build_post(progress)
    assert builder.op_builds[ops].post_built

    # error if builder doesn't define build_step
    @Builder.register(TestOp)  # pylint: disable=unused-variable
    class TestOpBuilder2(OpBuilder):
        def __init__(self, *_):
            super().__init__([], None, None)

    builder.build_pre(progress)
    with pytest.raises(BuildError):
        builder.build(progress)


@pytest.mark.parametrize("fail_fast", (True, False))
def test_custom_model(fail_fast):
    model = NengoModel(fail_fast=fail_fast)

    try:
        model.add_op(dummies.Op())
    except NotImplementedError:
        assert fail_fast
    else:
        assert not fail_fast
