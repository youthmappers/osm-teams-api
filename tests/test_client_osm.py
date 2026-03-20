"""Tests for _download_osm_users() and fetch_mapper_info_from_osm().

Covers: 403/429 abort, consecutive failure tracking, batch splitting,
User-Agent header, and dataframe enrichment.
"""
from __future__ import annotations

import json

import pytest
import responses
from pandas import DataFrame

from client import OSMTeamsError
from factories import make_osm_users_response

OSM_BASE = "https://openstreetmap.org/api/0.6/users.json"


class TestDownloadOsmUsersSuccess:
    @responses.activate
    def test_returns_dataframe_indexed_by_id(self, client):
        uids = [1, 2, 3]
        responses.add(
            responses.GET,
            OSM_BASE,
            json=make_osm_users_response(uids),
            status=200,
        )
        df = client._download_osm_users([str(u) for u in uids])
        assert list(df.index) == uids
        assert "display_name" in df.columns

    @responses.activate
    def test_sets_user_agent_header(self, client):
        responses.add(responses.GET, OSM_BASE, json=make_osm_users_response([1]), status=200)
        client._download_osm_users(["1"])
        assert "osm-teams-api" in responses.calls[0].request.headers["User-Agent"]

    @responses.activate
    def test_empty_input_returns_empty_dataframe(self, client):
        df = client._download_osm_users([])
        assert df.empty
        assert df.index.name == "id"
        assert len(responses.calls) == 0

    @responses.activate
    def test_deduplicates_uids(self, client):
        responses.add(responses.GET, OSM_BASE, json=make_osm_users_response([1]), status=200)
        df = client._download_osm_users(["1", "1", "1"])
        # Should only make one request with uid=1 (not three)
        assert len(responses.calls) == 1


class TestDownloadOsmUsersErrorHandling:
    @responses.activate
    def test_403_raises_immediately(self, client):
        responses.add(responses.GET, OSM_BASE, json={"error": "Forbidden"}, status=403)
        with pytest.raises(OSMTeamsError, match="HTTP 403"):
            client._download_osm_users(["1", "2"])
        assert len(responses.calls) == 1  # No retries or splits

    @responses.activate
    def test_429_raises_immediately(self, client):
        responses.add(responses.GET, OSM_BASE, json={"error": "Too Many Requests"}, status=429)
        with pytest.raises(OSMTeamsError, match="HTTP 429"):
            client._download_osm_users(["1", "2"])
        assert len(responses.calls) == 1

    @responses.activate
    def test_consecutive_failures_abort_after_threshold(self, client):
        # All batches fail with 500 — should abort after 3 consecutive failures
        responses.add(responses.GET, OSM_BASE, json={"error": "fail"}, status=500)
        with pytest.raises(OSMTeamsError, match="consecutive batches"):
            # Use enough UIDs to produce multiple batches (batch size is 100)
            # With 4 UIDs in one batch, the split produces batches of 2+2, then 1+1+1+1
            # The 3rd consecutive failure triggers the abort
            client._download_osm_users([str(i) for i in range(4)])

    @responses.activate
    def test_batch_split_on_non_blocking_failure(self, client):
        """When a batch of >1 fails with a non-blocking error, it splits and retries."""
        uids = [1, 2, 3, 4]
        full_url = OSM_BASE + "?users=1,2,3,4"
        half1_url = OSM_BASE + "?users=1,2"
        half2_url = OSM_BASE + "?users=3,4"

        # Full batch fails
        responses.add(responses.GET, full_url, json={"error": "fail"}, status=500)
        # Half batches succeed
        responses.add(responses.GET, half1_url, json=make_osm_users_response([1, 2]), status=200)
        responses.add(responses.GET, half2_url, json=make_osm_users_response([3, 4]), status=200)

        df = client._download_osm_users([str(u) for u in uids])
        assert sorted(df.index.tolist()) == uids
        assert len(responses.calls) == 3  # 1 fail + 2 halves

    @responses.activate
    def test_single_uid_failure_is_skipped(self, client):
        """A failure for a single UID is logged and skipped, not fatal."""
        # UID 1 succeeds, UID 2 fails (as single item after split)
        # We need to set up: batch [1,2] fails, batch [1] succeeds, batch [2] fails
        responses.add(
            responses.GET,
            OSM_BASE + "?users=1,2",
            json={"error": "fail"},
            status=500,
        )
        responses.add(
            responses.GET,
            OSM_BASE + "?users=1",
            json=make_osm_users_response([1]),
            status=200,
        )
        responses.add(
            responses.GET,
            OSM_BASE + "?users=2",
            json={"error": "fail"},
            status=500,
        )
        df = client._download_osm_users(["1", "2"])
        assert 1 in df.index
        assert 2 not in df.index

    @responses.activate
    def test_success_resets_failure_counter(self, client):
        """A successful batch resets the consecutive failure counter."""
        # We need batches to process sequentially. With 200 UIDs, we get 2 batches of 100.
        uids_batch1 = [str(i) for i in range(100)]
        uids_batch2 = [str(i) for i in range(100, 200)]

        url1 = OSM_BASE + "?users=" + ",".join(uids_batch1)
        url2 = OSM_BASE + "?users=" + ",".join(uids_batch2)

        # Batch 1 succeeds (resets counter), batch 2 fails then its splits succeed
        responses.add(
            responses.GET, url1,
            json=make_osm_users_response(list(range(100))),
            status=200,
        )
        # Batch 2 fails, splits into halves that succeed
        responses.add(responses.GET, url2, json={"error": "fail"}, status=500)

        half1_uids = list(range(100, 150))
        half2_uids = list(range(150, 200))
        half1_url = OSM_BASE + "?users=" + ",".join(str(u) for u in half1_uids)
        half2_url = OSM_BASE + "?users=" + ",".join(str(u) for u in half2_uids)
        responses.add(
            responses.GET, half1_url,
            json=make_osm_users_response(half1_uids),
            status=200,
        )
        responses.add(
            responses.GET, half2_url,
            json=make_osm_users_response(half2_uids),
            status=200,
        )

        df = client._download_osm_users([str(i) for i in range(200)])
        assert len(df) == 200


class TestFetchMapperInfoFromOsm:
    @responses.activate
    def test_enriches_dataframe_with_osm_columns(self, client):
        members_df = DataFrame(
            {"username": ["alice", "bob"]},
            index=[100, 200],
        )
        members_df.index.name = "id"

        responses.add(
            responses.GET,
            OSM_BASE,
            json=make_osm_users_response([100, 200]),
            status=200,
        )

        result = client.fetch_mapper_info_from_osm(members_df=members_df)
        assert "changesets" in result.columns
        assert "img" in result.columns
        # changesets should be extracted as integer count, not raw dict
        assert result.loc[100, "changesets"] == 1000
        # img should be extracted as URL string, not raw dict
        assert isinstance(result.loc[100, "img"], str)

    @responses.activate
    def test_osm_error_propagates(self, client):
        members_df = DataFrame({"username": ["alice"]}, index=[100])
        members_df.index.name = "id"

        responses.add(responses.GET, OSM_BASE, json={"error": "Forbidden"}, status=403)

        with pytest.raises(OSMTeamsError, match="HTTP 403"):
            client.fetch_mapper_info_from_osm(members_df=members_df)
