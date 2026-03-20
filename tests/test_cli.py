"""Tests for CLI entry points and exit behavior."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from pandas import DataFrame

from client import OSMTeamsError


class TestYouthmappersMainOsmError:
    def test_osm_error_exits_with_code_1(self):
        """When fetch_mapper_info_from_osm raises OSMTeamsError, main() should exit(1)."""
        mock_ym = MagicMock()
        mock_ym.fetch_mapper_info_from_osm.side_effect = OSMTeamsError("HTTP 403")

        # load_cached_data and build_mappers_dataframe must return valid dataframes
        mock_ym.load_cached_data.return_value = (DataFrame(), DataFrame())
        mock_ym.build_mappers_dataframe.return_value = DataFrame()

        with patch("youthmappers.YouthMappers", return_value=mock_ym):
            import youthmappers

            with pytest.raises(SystemExit) as exc_info:
                youthmappers.main(["--osm"])

            assert exc_info.value.code == 1
