"""Tests for the core _request(), _raise_for_status(), and _decode_response() methods."""
from __future__ import annotations

import responses

import pytest
from requests import ConnectionError as RequestsConnectionError

from client import OSMTeams, OSMTeamsError, OSMTeamsHTTPError


API_BASE = "https://mapping.team/api/"


class TestRequest:
    @responses.activate
    def test_success_json(self, client):
        responses.add(
            responses.GET,
            API_BASE + "teams/1",
            json={"id": 1, "name": "Team A"},
            status=200,
        )
        result = client._request("GET", "teams/1")
        assert result == {"id": 1, "name": "Team A"}

    @responses.activate
    def test_raises_http_error_on_4xx(self, client):
        responses.add(responses.GET, API_BASE + "teams/999", json={"error": "bad"}, status=400)
        with pytest.raises(OSMTeamsHTTPError, match="400"):
            client._request("GET", "teams/999")

    @responses.activate
    def test_raises_on_5xx_after_retries(self, client):
        # 500 is in the retry forcelist, so urllib3 retries and eventually raises
        # OSMTeamsError (from MaxRetryError), not OSMTeamsHTTPError
        responses.add(responses.GET, API_BASE + "teams/1", json={"error": "fail"}, status=500)
        with pytest.raises(OSMTeamsError):
            client._request("GET", "teams/1")

    @responses.activate
    def test_expected_statuses_suppresses_error(self, client):
        responses.add(responses.GET, API_BASE + "profiles/teams/1", body="Not Found", status=404)
        result = client._request("GET", "profiles/teams/1", expected_statuses=(404,))
        assert result == "Not Found"

    @responses.activate
    def test_network_error_wraps_in_osm_teams_error(self, client):
        responses.add(
            responses.GET,
            API_BASE + "teams/1",
            body=RequestsConnectionError("Connection refused"),
        )
        with pytest.raises(OSMTeamsError):
            client._request("GET", "teams/1")

    @responses.activate
    def test_invalid_json_raises_osm_teams_error(self, client):
        responses.add(
            responses.GET,
            API_BASE + "teams/1",
            body="not-json{",
            status=200,
            content_type="application/json",
        )
        with pytest.raises(OSMTeamsError, match="Invalid JSON"):
            client._request("GET", "teams/1")

    @responses.activate
    def test_returns_text_for_non_json(self, client):
        responses.add(
            responses.GET,
            API_BASE + "teams/1",
            body="<html>hello</html>",
            status=200,
            content_type="text/html",
        )
        result = client._request("GET", "teams/1")
        assert result == "<html>hello</html>"


class TestPrepareUrl:
    def test_relative_path(self, client):
        assert client._prepare_url("teams/1") == "https://mapping.team/api/teams/1"

    def test_absolute_url(self, client):
        url = "https://other.com/path"
        assert client._prepare_url(url) == url

    def test_leading_slash_stripped(self, client):
        assert client._prepare_url("/teams/1") == "https://mapping.team/api/teams/1"
