from __future__ import absolute_import, division, print_function

import pytest
import pyro

def pytest_collection_modifyitems(items):
    for item in items:
        if "init" not in item.keywords:
            item.add_marker(pytest.mark.init(rng_seed=123))

def pytest_runtest_setup(item):
    test_initialize_marker = item.get_closest_marker("init")
    if test_initialize_marker:
        rng_seed = test_initialize_marker.kwargs["rng_seed"]
        pyro.set_rng_seed(rng_seed)
