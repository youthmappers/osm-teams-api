"""Tests for get_team_attributes() defensive parsing of varied API response shapes."""
from __future__ import annotations

import responses

from client import OSMTeams

API_BASE = "https://mapping.team/api/"


class TestGetTeamAttributes:
    @responses.activate
    def test_list_format(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/1",
            json=[{"name": "University", "value": "MIT"}, {"name": "City", "value": "Boston"}],
            status=200,
        )
        result = client.get_team_attributes(1)
        assert result == {"University": "MIT", "City": "Boston"}

    @responses.activate
    def test_dict_with_attributes_key(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/2",
            json={"attributes": [{"name": "Country", "value": "USA"}]},
            status=200,
        )
        result = client.get_team_attributes(2)
        assert result == {"Country": "USA"}

    @responses.activate
    def test_error_response_returns_empty(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/3",
            json={"statusCode": 404, "error": "Not Found", "message": "No attributes"},
            status=200,
        )
        result = client.get_team_attributes(3)
        assert result == {}

    @responses.activate
    def test_404_returns_empty(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/4",
            body="Not Found",
            status=404,
        )
        result = client.get_team_attributes(4)
        assert result == {}

    @responses.activate
    def test_empty_list_returns_empty(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/5",
            json=[],
            status=200,
        )
        result = client.get_team_attributes(5)
        assert result == {}

    @responses.activate
    def test_items_without_name_are_skipped(self, client):
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/6",
            json=[{"name": "Valid", "value": "yes"}, {"value": "orphan"}, "not-a-dict"],
            status=200,
        )
        result = client.get_team_attributes(6)
        assert result == {"Valid": "yes"}

    @responses.activate
    def test_flat_dict_format(self, client):
        """Some responses return a flat dict of key-value pairs."""
        responses.add(
            responses.GET,
            API_BASE + "profiles/teams/7",
            json={"University": "Stanford", "City": "Palo Alto"},
            status=200,
        )
        result = client.get_team_attributes(7)
        assert result == {"University": "Stanford", "City": "Palo Alto"}
