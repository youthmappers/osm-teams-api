import argparse
import logging
import os
from pathlib import Path

from client import (
    LOG_LEVELS,
    OSMTeams,
    build_parser,
    load_teams_dataframe,
    save_members_dataframe,
    save_teams_dataframe,
)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(args=argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    client = OSMTeams(organization_id=args.id)

    teams = client.get_all_organization_teams()
    save_teams_dataframe(teams, Path(args.teams_cache))

    members_df = client.get_all_organization_members()
    save_members_dataframe(members_df, Path(args.members_cache))

    loaded_teams = load_teams_dataframe(Path(args.teams_cache))
    # print(loaded_teams)

    if args.osm:
        logger.info("Enriching members with OpenStreetMap profile information")
        members_df = client.fetch_mapper_info_from_osm(members_df=members_df)
        members_df.to_csv("enriched_members.csv")

        print(members_df)


if __name__ == "__main__":
    main()
