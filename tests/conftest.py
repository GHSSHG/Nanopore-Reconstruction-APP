"""Shared fixtures."""

import pytest

from nanorecon.model_config import PROFILE, CodecProfile

from .helpers import LookupEngine


@pytest.fixture(scope="session")
def profile() -> CodecProfile:
    return PROFILE


@pytest.fixture
def lookup_engine_factory(profile):
    def make(batch_size: int = 4):
        return LookupEngine(batch_size, profile)
    return make
