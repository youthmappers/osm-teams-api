from __future__ import annotations

import pytest
import responses

from client import OSMTeams


@pytest.fixture
def client():
    """OSMTeams client configured for testing — no token, short timeout, single worker."""
    return OSMTeams(organization_id=99, token=None, per_page=10, max_workers=1, timeout=5.0)


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps
