# Licensed under a 3-clause BSD style license - see LICENSE.rst
from dataclasses import dataclass

import numpy as np
import pytest

import chandra_aca.centroid_resid as cacr


@dataclass
class MyTestTime(cacr.TimeBase):
    vals: np.ndarray
    meta: object = None

    def __post_init__(self):
        super().__post_init__()
        self.vals = np.asarray(self.vals)


def test_post_init_casts_times_to_float():
    tb = MyTestTime([1, 2, 3], [10, 20, 30])
    assert isinstance(tb.times, np.ndarray)
    assert tb.times.dtype == np.float64


def test_getitem_with_int_slice():
    tb = MyTestTime([1, 2, 3, 4], [10, 20, 30, 40], meta="foo")
    tb2 = tb[1:3]
    np.testing.assert_array_equal(tb2.times, [2, 3])
    np.testing.assert_array_equal(tb2.vals, [20, 30])
    assert tb2.meta == "foo"


def test_getitem_with_float_slice():
    tb = MyTestTime([10, 20, 30, 40], [100, 200, 300, 400], meta="bar")
    tb2 = tb[0.0:10.0]  # Should select only the first element
    np.testing.assert_array_equal(tb2.times, [10])
    np.testing.assert_array_equal(tb2.vals, [100])
    assert tb2.meta == "bar"

    tb2 = tb[0.0:10.01]  # Should select two first elements
    np.testing.assert_array_equal(tb2.times, [10, 20])
    np.testing.assert_array_equal(tb2.vals, [100, 200])


def test_getitem_with_negative_float_slice():
    tb = MyTestTime([10, 20, 30, 40], [100, 200, 300, 400])
    tb2 = tb[-10.0:None]  # Should select last two element
    np.testing.assert_array_equal(tb2.times, [30, 40])
    np.testing.assert_array_equal(tb2.vals, [300, 400])

    tb2 = tb[-9.99:None]  # Should select last element
    np.testing.assert_array_equal(tb2.times, [40])
    np.testing.assert_array_equal(tb2.vals, [400])


def test_getitem_with_int_index():
    tb = MyTestTime([1, 2, 3], [10, 20, 30])
    tb2 = tb[1]
    assert isinstance(tb2, MyTestTime)
    np.testing.assert_array_equal(tb2.times, 2)
    np.testing.assert_array_equal(tb2.vals, 20)


def test_getitem_with_array_index():
    tb = MyTestTime([1, 2, 3], [10, 20, 30])
    tb2 = tb[[0, 2]]
    np.testing.assert_array_equal(tb2.times, [1, 3])
    np.testing.assert_array_equal(tb2.vals, [10, 30])


def test_getitem_with_nonarray_attribute():
    tb = MyTestTime([1, 2], [10, 20], meta={"foo": 1})
    tb2 = tb[1]
    assert tb2.meta == {"foo": 1}


def test_getitem_with_mismatched_length_attribute():
    @dataclass
    class TB(MyTestTime):
        other: object = None

    tb = TB([1, 2, 3], [10, 20, 30], other=[1, 2])
    tb2 = tb[1:3]
    # 'other' is not the same length as times, so should be passed through unchanged
    assert tb2.other == [1, 2]


@pytest.mark.parametrize("item", [slice(0, "foo"), slice("bar", 2.0), slice(0.0, 2)])
def test_get_slice_or_item_invalid_types(item):
    tb = MyTestTime([1, 2, 3], [10, 20, 30])
    with pytest.raises(
        ValueError,
        match="slice start, stop must be all int/None or all float/None",
    ):
        tb._get_slice_or_item(item)
