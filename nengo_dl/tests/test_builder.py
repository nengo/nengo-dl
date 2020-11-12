# pylint: disable=missing-docstring

import pytest
from nengo.exceptions import BuildError

from nengo_dl.builder import Builder, NengoModel, OpBuilder
from nengo_dl.tests import dummies
from nengo_dl.utils import NullProgressBar


def test_custom_builder():
    # pylint: disable=unused-variable

    class TestOp:
        sets = None
        incs = None
        reads = None
        updates = None

    ops = (TestOp(),)
    progress = NullProgressBar()

    # error if no builder registered
    with pytest.raises(BuildError, match="No registered builder"):
        Builder([ops])

    # warning if builder doesn't subclass OpBuilder
    with pytest.warns(UserWarning):

        @Builder.register(TestOp)
        class TestOpBuilder0:
            pass

    # warning when overwriting a registered builder
    with pytest.warns(UserWarning):

        @Builder.register(TestOp)
        class TestOpBuilder(OpBuilder):
            pre_built = False
            post_built = False

            def build_pre(self, signals, config):
                super().build_pre(signals, config)
                self.pre_built = True

            def build_step(self, signals):
                assert self.pre_built
                assert not self.post_built

                return 0, 1

            def build_post(self, signals):
                self.post_built = True

    builder = Builder([ops])

    builder.build_pre(signals=None, config=None, progress=progress)

    result = builder.build_step(signals=None, progress=progress)

    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 1

    builder.build_post(signals=None, progress=progress)
    assert builder.op_builds[ops].post_built

    # error if builder doesn't define build_step
    @Builder.register(TestOp)
    class TestOpBuilder2(OpBuilder):
        pass

    builder = Builder([ops])
    builder.build_pre(signals=None, config=None, progress=progress)
    with pytest.raises(BuildError, match="must implement a `build_step` function"):
        builder.build_step(signals=None, progress=progress)


@pytest.mark.parametrize("fail_fast", (True, False))
def test_custom_model(fail_fast):
    model = NengoModel(fail_fast=fail_fast)

    try:
        model.add_op(dummies.Op())
    except NotImplementedError:
        assert fail_fast
    else:
        assert not fail_fast
