from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
from shapely import to_wkt

import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame, Timestamp

from client import (
    OSMTeams,
    build_parser,
)

try:  # Optional mixin for Google-specific features
    from youthmappers_google import GoogleSheetsMixin
except ImportError:  # pragma: no cover - mixin not available

    class GoogleSheetsMixin:  # type: ignore
        """Fallback mixin when the google extras are not installed."""

        pass


YM_COLUMNS = [
    "username",
    "Name",
    "Gender",
    "Major or Degree Concentration",
    "Graduation Date",
    "Hometown and Country",
    "Role / Position",
    "team_id",
    "all_teams",
    "source",
    "alumni_date",
]
PII_COLUMNS = ["Year Born", "Email"]
BADGE_COLUMNS = [
    "Steering Committee",
    "Regional Ambassador",
    "Alumni",
    "Mentor / Faculty Advisor",
]


class YouthMappers(GoogleSheetsMixin, OSMTeams):
    def __init__(self) -> None:
        super().__init__(token=os.getenv("OSM_TEAMS_ACCESS_TOKEN"), organization_id=1)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.chapters_id = "1ID2FWaqRdTMF32obGS3Utx5D2F4MPeQ6"
        self.members_id = "1tY2KHbVUwqHYxZVVb4dZRsMe8oFSLdMJ"

        creds_b64 = os.getenv("YGL_GOOGLE_CREDENTIALS")
        self.creds = json.loads(base64.b64decode(creds_b64)) if creds_b64 else None

    def build_mappers_dataframe(
        self,
        members_df: DataFrame,
        teams_df: DataFrame,
    ) -> DataFrame:
        if "member_uids" not in teams_df.columns:
            raise ValueError("Teams dataframe must include a 'member_uids' column")

        working = teams_df.reset_index()
        if "team_id" not in working.columns:
            if "index" in working.columns:
                working = working.rename(columns={"index": "team_id"})
            elif "id" in working.columns:
                working = working.rename(columns={"id": "team_id"})
            else:
                raise ValueError(
                    "Teams dataframe must contain an index or column that can serve as 'team_id'."
                )
        membership = (
            working[["team_id", "member_uids"]]
            .explode("member_uids")
            .dropna(subset=["member_uids"])
        )

        lookup = membership.groupby("member_uids").agg(
            team_id=("team_id", "min"),
            all_teams=("team_id", list),
        )

        mappers = members_df.copy()
        mappers.index.name = mappers.index.name or "uid"
        mappers = mappers.join(lookup, how="left")
        mappers["source"] = "OSM Teams"

        if "Alumni" in mappers.columns:
            mappers["alumni_date"] = mappers["Alumni"].apply(
                lambda payload: (
                    Timestamp(payload.get("assigned_at"))
                    if isinstance(payload, dict) and payload.get("assigned_at")
                    else pd.NaT
                )
            )

        return mappers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YouthMappers utility on top of the OSM Teams API")
    parser = build_parser(parser)

    # Add YM parsing arguments here if needed
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Fetch latest teams and members from OSM Teams API",
    )
    
    parser.add_argument(
        "--conflate",
        action="store_true",
        help="Conflate mappers with data from previous Master list from Google Drive",
    )

    args = parser.parse_args(args=argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    ym = YouthMappers()

    teams_cache = Path("chapters.geojson")
    members_cache = Path("members.json")

    if args.teams:
        logger.info("Fetching latest teams and members from OSM Teams")
        chapters_df = ym.fetch_and_cache_teams(cache_path=teams_cache)
        members_df = ym.fetch_and_cache_members(cache_path=members_cache)

    try:
        members_df, chapters_df = ym.load_cached_data(
            members_path=members_cache,
            teams_path=teams_cache,
        )
    except FileNotFoundError as exc:
        logger.error("%s not found. Run with --teams first to prime the cache.", exc)
        raise SystemExit(1) from exc

    # Build mappers dataframe
    mappers_df = ym.build_mappers_dataframe(
        members_df=members_df,
        teams_df=chapters_df,
    )

    if args.conflate:
        logger.info("Conflating mappers with previous Master list from Google Drive")
        mappers_df = ym.fetch_previous_master_list_and_conflate(mappers_df=mappers_df)
        mappers_df.to_parquet("youthmappers-conflated.parquet")

    if args.osm:
        logger.info("Enriching members with OpenStreetMap profile information")
        mappers_df = ym.fetch_mapper_info_from_osm(members_df=mappers_df)

    mappers_df.to_parquet("tmp_mappers.parquet")

    output = GeoDataFrame(
        mappers_df.merge(
            chapters_df[["name", "University", "City", "Country", "geometry"]].rename(
                columns={
                    "name": "chapter",
                    "University": "university",
                    "City": "city",
                    "Country": "country",
                }
            ),
            left_on="team_id",
            right_index=True,
        )
    )
    output = output.reset_index().rename(columns={"index": "uid"})

    # Clean up some of the structured columns:
    output["alumni"] = output["Alumni"].apply(
        lambda d: (
            pd.to_datetime(d["assigned_at"]).date()
            if isinstance(d, dict) and "assigned_at" in d
            else pd.NaT
        )
    )
    output["ymsc"] = output["Steering Committee"].apply(
        lambda d: (
            pd.to_datetime(d["assigned_at"]).date()
            if isinstance(d, dict) and "assigned_at" in d
            else pd.NaT
        )
    )
    output["regional_ambassador"] = output["Regional Ambassador"].apply(
        lambda d: (
            pd.to_datetime(d["assigned_at"]).date()
            if isinstance(d, dict) and "assigned_at" in d
            else pd.NaT
        )
    )
    output["mentor_faculty_advisor"] = output["Mentor / Faculty Advisor"].apply(
        lambda d: (
            pd.to_datetime(d["assigned_at"]).date()
            if isinstance(d, dict) and "assigned_at" in d
            else pd.NaT
        )
    )

    # Replace `gender` with cleaned up version
    output["gender"] = output["Gender"]

    # Just a date for account_created
    output["account_created"] = output["account_created"].apply(lambda d: d.date())

    output[
        [
            "uid",
            "username",
            "gender",
            "team_id",
            "alumni",
            "ymsc",
            "regional_ambassador",
            "mentor_faculty_advisor",
            "chapter",
            "university",
            "city",
            "country",
            "account_created",
            "description",
            "img",
            "changesets",
            "company",
            "geometry",
        ]
    ].to_parquet("youthmappers.zstd.parquet", compression="zstd")


if __name__ == "__main__":
    main()
