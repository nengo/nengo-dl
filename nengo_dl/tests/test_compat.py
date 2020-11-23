# pylint: disable=missing-docstring

from nengo_dl import compat


def test_ordered_set():
    my_tuple = ("c", "b", "a")
    my_frozen_set = compat.FrozenOrderedSet(my_tuple)

    # order is preserved
    assert tuple(my_frozen_set) == my_tuple

    # test member functions
    for x in my_tuple:
        assert x in my_frozen_set

    assert "z" not in my_frozen_set

    assert len(my_frozen_set) == len(my_tuple)

    # hashable
    assert {my_frozen_set: "val"}[my_frozen_set] == "val"
