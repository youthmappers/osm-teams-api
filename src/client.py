from __future__ import annotations

import json
import logging
import argparse
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geopandas import GeoDataFrame
from pandas import DataFrame, read_json, to_datetime
from requests import RequestException, Response, Session
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from urllib3.util.retry import Retry

LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

MEMBERS_CACHE = Path("members.json")
TEAMS_CACHE = Path("teams.geojson")


class OSMTeamsError(Exception):
    """Base exception for OSM Teams client errors."""


class OSMTeamsHTTPError(OSMTeamsError):
    """Raised when the API responds with a non-successful HTTP status."""


def save_members_dataframe(df: DataFrame, path: str | Path = MEMBERS_CACHE) -> None:
    Path(path).write_text(df.to_json(orient="split"))


def load_members_dataframe(path: str | Path = MEMBERS_CACHE) -> DataFrame:
    return read_json(path, orient="split")


def save_teams_dataframe(df: DataFrame, path: str | Path = TEAMS_CACHE) -> None:
    Path(path).write_text(df.to_json())


def load_teams_dataframe(path: str | Path = TEAMS_CACHE) -> DataFrame:
    path = Path(path)
    with path.open() as fh:
        payload = json.load(fh)

    features = payload.get("features")
    if features:
        return GeoDataFrame.from_features(features)

    # Fallback to plain dataframe
    return read_json(path)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="OSM Teams CLI for Organizations")
    parser.add_argument("--id", type=int, default=5, help="Organization ID")
    parser.add_argument(
        "--osm", action="store_true", help="Enrich cached members with data from osm.org"
    )
    parser.add_argument(
        "--parquet", action="store_true", help="Write mappers.parquet alongside the CSV output"
    )
    parser.add_argument(
        "--members-cache",
        default=str(MEMBERS_CACHE),
        help="Path to the cached members JSON (default: members.json)",
    )
    parser.add_argument(
        "--teams-cache",
        default=str(TEAMS_CACHE),
        help="Path to the cached teams GeoJSON (default: chapters.geojson)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO)",
    )
    return parser


@dataclass
class RetryConfig:
    total: int = 3
    backoff_factor: float = 0.3
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504)
    allowed_methods: Sequence[str] = ("GET", "PUT", "POST")


class OSMTeams:
    """Thin, extensible client for the OSM Teams API."""

    def __init__(
        self,
        organization_id: int,
        token: str | None = None,
        per_page: int = 50,
        debug: bool = False,
        max_workers: int = 8,
        timeout: float = 30.0,
        retry_config: RetryConfig | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
    ) -> None:
        self.API_URL = "https://mapping.team/api/"
        self.ORG = organization_id
        self.PER_PAGE = per_page
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self._executor_factory = executor_factory or ThreadPoolExecutor

        self.debug = debug
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(
            "Initializing OSMTeams client (org=%s, per_page=%s, max_workers=%s, timeout=%s, debug=%s)",
            self.ORG,
            self.PER_PAGE,
            self.max_workers,
            self.timeout,
            self.debug,
        )

        headers = None
        if token:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }

        self.session = self._configure_session(headers=headers)

    @contextmanager
    def _session_scope(self):
        clone = self._configure_session(headers=dict(self.session.headers))
        try:
            yield clone
        finally:
            clone.close()

    def _configure_session(
        self,
        *,
        headers: dict[str, str] | None = None,
    ) -> Session:
        session = Session()
        if headers:
            session.headers.update(headers)
        retry = Retry(
            total=self.retry_config.total,
            backoff_factor=self.retry_config.backoff_factor,
            status_forcelist=self.retry_config.status_forcelist,
            allowed_methods=[method.upper() for method in self.retry_config.allowed_methods],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _prepare_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.API_URL}{path_or_url.lstrip('/')}"

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        session: Session | None = None,
        params: dict[str, Any] | None = None,
        data: Any | None = None,
        json_payload: Any | None = None,
        timeout: float | None = None,
        expected_statuses: Sequence[int] | None = None,
    ) -> Any:
        url = self._prepare_url(path_or_url)
        session = session or self.session
        timeout = timeout or self.timeout
        self.logger.debug("Request %s %s", method.upper(), url)
        try:
            response = session.request(
                method.upper(),
                url,
                params=params,
                data=data,
                json=json_payload,
                timeout=timeout,
            )
        except RequestException as exc:
            self.logger.exception("Network error during %s %s", method, url)
            raise OSMTeamsError(str(exc)) from exc

        self._raise_for_status(response, method, url, expected_statuses)
        return self._decode_response(response)

    def _raise_for_status(
        self,
        response: Response,
        method: str,
        url: str,
        expected_statuses: Sequence[int] | None = None,
    ) -> None:
        expected_statuses = tuple(expected_statuses or ())
        if response.ok or response.status_code in expected_statuses:
            return
        self.logger.error("API error %s %s -> %s", method.upper(), url, response.status_code)
        raise OSMTeamsHTTPError(
            f"{method.upper()} {url} failed with {response.status_code}: {response.text}"
        )

    def _decode_response(self, response: Response) -> Any:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return response.json()
            except ValueError as exc:
                self.logger.exception("Failed to decode JSON response")
                raise OSMTeamsError("Invalid JSON response") from exc
        return response.text

    def _handle_pages(
        self,
        pagination,
        url,
        per_page: int | None = None,
        max_pages: int | None = None,
        session=None,
    ):
        session = session or self.session
        total_pages = pagination.get("lastPage") or 1

        effective_limit = max_pages or total_pages
        if self.debug:
            effective_limit = min(effective_limit, 2)
        target_last_page = min(total_pages, max(1, effective_limit))

        if target_last_page <= 1:
            self.logger.debug("Single-page response for %s; skipping pagination requests", url)
            return []

        limit_suffix = f" of {total_pages}" if target_last_page < total_pages else ""
        self.logger.info(
            "Handling pagination for %s (pages=%s%s)",
            url,
            target_last_page,
            limit_suffix,
        )

        data = []
        for page in range(2, target_last_page + 1):
            page_url = url + "?page=" + str(page) + (f"&perPage={per_page}" if per_page else "")
            self.logger.debug("Requesting page %s -> %s", page, page_url)
            res = self._request("GET", page_url, session=session)
            data.append(res)
            marker = str(page) if ((page - 1) % 10 == 0 or page == target_last_page) else "."
            print(marker, end="", flush=True)

        print("")

        return data

    def get_team(self, team: int, session: Session | None = None) -> dict[str, Any]:
        """Return metadata for a single team."""
        return self._request("GET", f"teams/{team}", session=session)

    def get_team_members(self, team: int, session: Session | None = None):
        session = session or self.session
        _page1 = self._request("GET", f"teams/{team}/members", session=session)

        if "members" in _page1:
            data = _page1.get("members").get("data")

            if "pagination" in _page1.get("members"):
                pagination = _page1.get("members").get("pagination")

                remaining_pages = self._handle_pages(
                    pagination, self.API_URL + f"teams/{team}/members", session=session
                )

                for page in remaining_pages:
                    data += page.get("members").get("data")

                if not self.debug:
                    assert len(data) == pagination.get(
                        "total"
                    ), "Pagination returned incorrect number of team members"

        return data

    # Not paginated
    def get_team_moderators(self, team: int, session: Session | None = None):
        session = session or self.session
        return self._request("GET", f"teams/{team}/moderators", session=session)

    def get_team_attributes(self, team: int, session: Session | None = None):
        session = session or self.session
        res = self._request(
            "GET",
            f"profiles/teams/{team}",
            session=session,
            expected_statuses=(404,),
        )
        if not res or isinstance(res, str):
            return {}

        try:
            if isinstance(res, list):
                return {
                    item.get("name"): item.get("value")
                    for item in res
                    if isinstance(item, dict) and item.get("name") is not None
                }

            if isinstance(res, dict):
                if {"statusCode", "error"}.issubset(res.keys()):
                    return {}

                attributes_payload = res.get("attributes") if "attributes" in res else res
                if isinstance(attributes_payload, list):
                    return {
                        item.get("name"): item.get("value")
                        for item in attributes_payload
                        if isinstance(item, dict) and item.get("name") is not None
                    }
                if isinstance(attributes_payload, dict):
                    return {
                        key: value
                        for key, value in attributes_payload.items()
                        if not isinstance(key, (int, float))
                        and key not in {"statusCode", "error", "message"}
                    }

            self.logger.warning(
                "Unexpected team attributes payload for team %s: %s", team, type(res)
            )
            return {}
        except Exception:
            self.logger.exception("Failed to parse team attributes response for team %s", team)
            return {}

    def get_all_organization_teams(
        self,
        per_page: int = 50,
    ) -> DataFrame:
        """Fetch and cache the base teams dataframe for the configured organization."""

        self.logger.info("Fetching all organization teams")

        url = self.API_URL + "organizations/" + str(self.ORG) + "/teams"

        _page1 = self._request("GET", url + f"?page=1&perPage={per_page}")

        if "data" in _page1:
            data = _page1.get("data")

            if "pagination" in _page1:
                pagination = _page1.get("pagination")

                self.logger.info(
                    "Paginating %s pages for %s teams",
                    f"{pagination.get('lastPage'):,}",
                    f"{pagination.get('total'):,}",
                )

                remaining_pages = self._handle_pages(pagination, url, per_page=per_page)

                for page in remaining_pages:
                    data += page.get("data")

                assert len(data) == pagination.get(
                    "total"
                ), f"Pagination returned incorrect number of teams: {len(data)} / {pagination.get('total')}"

        df = DataFrame(data).set_index("id")
        try:
            geometry = df["location"].apply(lambda loc: shape(json.loads(loc)) if loc else None)
        except Exception:
            self.logger.warning(
                "Failed to convert team locations to geometries; returning plain DataFrame"
            )
            geometry = None

        if geometry is not None:
            df = GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            df.drop("location", axis=1, inplace=True)

        self._teams = df
        return df

    def fetch_team_details(
        self,
        *,
        attributes: bool = True,
        members: bool = True,
        join_link: bool = True,
        teams_df: DataFrame | None = None,
    ) -> DataFrame:
        """Fetch optional team enrichments in parallel and join them to the teams dataframe."""

        if not (attributes or members or join_link):
            self.logger.debug("No team enrichments requested; returning cached dataframe")
            cached = teams_df if teams_df is not None else getattr(self, "_teams", None)
            if cached is None:
                raise ValueError("No teams dataframe available. Fetch teams first.")
            return cached

        df_source = teams_df if teams_df is not None else getattr(self, "_teams", None)
        if df_source is None:
            raise ValueError(
                "No teams dataframe available to enrich. Fetch teams first or provide one explicitly."
            )

        df = df_source.copy()
        team_ids = list(df.index)
        if not team_ids:
            self.logger.warning("No teams available for enrichment; returning dataframe unchanged")
            return df

        details = {"attributes": [], "members": [], "join_links": []}
        max_workers = max(1, min(self.max_workers, len(team_ids)))

        self.logger.debug("Enriching %s teams with %s workers", len(team_ids), max_workers)

        with self._executor_factory(max_workers=max_workers) as executor:  # type: ignore[arg-type]
            future_map = {
                executor.submit(
                    self._collect_team_details,
                    team_id,
                    attributes,
                    members,
                    join_link,
                ): team_id
                for team_id in team_ids
            }

            processed = 0
            for future in as_completed(future_map):
                team_id = future_map[future]
                try:
                    detail = future.result()
                except Exception as exc:
                    self.logger.exception("Failed to fetch data for team %s: %s", team_id, exc)
                    continue

                if attributes and detail.get("attributes"):
                    details["attributes"].append(detail["attributes"])
                if members and detail.get("members"):
                    details["members"].append(detail["members"])
                if join_link and detail.get("join_link"):
                    details["join_links"].append(detail["join_link"])

                processed += 1
                if processed % 10 == 0 or processed == len(team_ids):
                    self.logger.info("Team enrichment progress: %s/%s", processed, len(team_ids))

        if attributes and details["attributes"]:
            attr_df = DataFrame(details["attributes"]).set_index("id")
            df = df.join(attr_df, how="left")

        if members and details["members"]:
            members_df = DataFrame(details["members"]).set_index("id")
            df = df.join(members_df, how="left")

        if join_link and details["join_links"]:
            join_df = (
                DataFrame(details["join_links"])
                .set_index("team_id")[["id"]]
                .rename(columns={"id": "join_link"})
            )
            df = df.join(join_df, how="left")

        self._teams = df
        return df

    def _collect_team_details(self, team_id, attributes, members, join_link):
        payload = {}
        with self._session_scope() as session:
            if members:
                res1 = self.get_team_members(team_id, session=session)
                member_uids = set([x.get("id") for x in res1]) if res1 else set()
                res2 = self.get_team_moderators(team_id, session=session)
                moderator_uids = set([x.get("osm_id") for x in res2]) if res2 else set()
                payload["members"] = {
                    "id": team_id,
                    "member_uids": list(member_uids),
                    "moderator_uids": list(moderator_uids),
                }

            if attributes:
                team_attributes = self.get_team_attributes(team_id, session=session)
                if team_attributes is not None:
                    team_attributes["id"] = team_id
                    payload["attributes"] = team_attributes

            if join_link:
                join_link_data = self.get_join_link(team_id, session=session)
                if join_link_data:
                    payload["join_link"] = join_link_data[0]

        return payload

    def get_all_organization_members(
        self,
        org_id: int | None = None,
        *,
        max_pages: int | None = None,
    ) -> DataFrame:
        """Fetch and cache the base dataframe for all organization members, optionally limiting pagination."""

        org_identifier = org_id or self.ORG
        self.logger.info("Fetching all members for org %s", org_identifier)

        url = f"{self.API_URL}organizations/{org_identifier}/members"
        _page1 = self._request("GET", url)

        data = _page1.get("data", [])

        if "pagination" in _page1:
            pagination = _page1["pagination"]
            self.logger.info(
                "Paginating %s pages for %s users",
                f"{pagination.get('lastPage'):,}",
                f"{pagination.get('total'):,}",
            )

            remaining_pages = self._handle_pages(pagination, url, max_pages=max_pages)
            for page in remaining_pages:
                data += page.get("data", [])

            if not self.debug and not max_pages:
                assert len(data) == pagination.get(
                    "total"
                ), "Pagination returned incorrect number of members for organization"

        if not data:
            df = DataFrame(columns=["uid", "username"]).set_index("uid")
        else:
            df = DataFrame(data)
            df.rename(columns={"id": "uid", "name": "username"}, inplace=True)
            df.set_index("uid", inplace=True)

        self._members = df
        return df

    def fetch_member_details(
        self,
        *,
        attributes: bool = True,
        badges: bool = True,
        members_df: DataFrame | None = None,
    ) -> DataFrame:
        """Fetch optional member enrichments in parallel and join them to the members dataframe."""

        if not (attributes or badges):
            cached = members_df if members_df is not None else getattr(self, "_members", None)
            if cached is None:
                raise ValueError("No members dataframe available. Fetch members first.")
            return cached

        df_source = members_df if members_df is not None else getattr(self, "_members", None)
        if df_source is None:
            raise ValueError(
                "No members dataframe available to enrich. Fetch members first or provide one explicitly."
            )

        df = df_source.copy()
        uids = list(df.index)
        if not uids:
            self.logger.warning(
                "No members available for enrichment; returning dataframe unchanged"
            )
            return df

        details = {"attributes": [], "badges": []}
        max_workers = max(1, min(self.max_workers, len(uids)))

        self.logger.debug("Enriching %s members with %s workers", len(uids), max_workers)

        with self._executor_factory(max_workers=max_workers) as executor:  # type: ignore[arg-type]
            future_map = {
                executor.submit(
                    self._collect_member_details,
                    uid,
                    attributes,
                    badges,
                ): uid
                for uid in uids
            }

            processed = 0
            for future in as_completed(future_map):
                uid = future_map[future]
                try:
                    detail = future.result()
                except Exception as exc:
                    self.logger.exception("Failed to fetch data for uid %s: %s", uid, exc)
                    continue

                if attributes and detail.get("attributes"):
                    details["attributes"].append(detail["attributes"])
                if badges and detail.get("badges"):
                    details["badges"].append(detail["badges"])

                processed += 1
                if processed % 100 == 0 or processed == len(uids):
                    self.logger.info("Member enrichment progress: %s/%s", processed, len(uids))

        if attributes and details["attributes"]:
            attr_df = DataFrame(details["attributes"]).set_index("uid")
            df = df.join(attr_df, how="left")

        if badges and details["badges"]:
            badges_df = DataFrame(details["badges"]).set_index("uid")
            df = df.join(badges_df, how="left")

        self._members = df
        return df

    def _collect_member_details(self, uid, want_attributes, want_badges):
        payload = {}
        with self._session_scope() as session:
            if want_attributes:
                user_attributes = self.get_org_user_attributes(uid, session=session)
                if user_attributes is not None:
                    user_attributes["uid"] = uid
                    payload["attributes"] = user_attributes
            if want_badges:
                user_badges = self.get_user_badges(uid, session=session)
                if user_badges is not None:
                    user_badges["uid"] = uid
                    payload["badges"] = user_badges
        return payload

    def get_org_user_attributes(self, uid: int, session: Session | None = None):
        """Return organization-specific attributes for a user, or None if absent."""
        session = session or self.session
        res = self._request(
            "GET",
            f"profiles/organizations/{self.ORG}/{uid}",
            session=session,
            expected_statuses=(404,),
        )
        if not res or isinstance(res, str):
            return None

        try:
            if isinstance(res, list):
                return {
                    item.get("name"): item.get("value")
                    for item in res
                    if isinstance(item, dict) and item.get("name") is not None
                } or None

            if isinstance(res, dict):
                if {"statusCode", "error"}.issubset(res.keys()):
                    return None

                payload = res.get("attributes") if "attributes" in res else res
                if isinstance(payload, list):
                    return {
                        item.get("name"): item.get("value")
                        for item in payload
                        if isinstance(item, dict) and item.get("name") is not None
                    } or None
                if isinstance(payload, dict):
                    return {
                        key: value
                        for key, value in payload.items()
                        if not isinstance(key, (int, float))
                        and key not in {"statusCode", "error", "message"}
                    } or None

            self.logger.warning("Unexpected org attribute payload for uid %s: %s", uid, type(res))
            return None
        except Exception:
            self.logger.exception("Failed to parse org attributes for uid %s", uid)
            return None

    def get_user_badges(self, uid: int, session: Session | None = None):
        """Return organization badges for a given user."""
        session = session or self.session
        res = self._request(
            "GET",
            f"user/{uid}/badges",
            session=session,
            expected_statuses=(404,),
        )
        if isinstance(res, str):
            return None
        badges_payload = res.get("badges", [])
        if badges_payload:
            return {b.get("name"): json.loads(json.dumps(b)) for b in badges_payload}
        return None

    def fetch_mapper_info_from_osm(
        self,
        members_df: DataFrame | None = None,
        *,
        timeout: float | None = None,
        session: Session | None = None,
    ) -> DataFrame:
        """Fetch OSM user metadata and enrich the provided (or cached) members dataframe.

        Expects a members dataframe with OSM UIDs as the index.
        """

        df_source = members_df if members_df is not None else getattr(self, "_members", None)
        if df_source is None:
            raise ValueError(
                "No members dataframe available. Provide one or call get_all_organization_members first."
            )

        osm_df = self._download_osm_users(
            [str(i) for i in df_source.index], timeout=timeout, session=session
        )

        for col in ["changesets", "traces"]:
            if col in osm_df.columns:
                osm_df[col] = osm_df[col].apply(
                    lambda payload: payload.get("count") if isinstance(payload, dict) else 0
                )

        if "img" in osm_df.columns:
            osm_df["img"] = osm_df["img"].apply(
                lambda payload: payload.get("href") if isinstance(payload, dict) else None
            )

        if "account_created" in osm_df.columns:
            osm_df["account_created"] = to_datetime(osm_df["account_created"], errors="coerce")

        if osm_df.empty:
            self.logger.warning("OSM user lookup returned no data; dataframe left unchanged")
            return df_source

        enriched = df_source.join(osm_df, how="left")

        # Store the enriched dataframe on the class.
        self._members = enriched
        return enriched

    def _download_osm_users(
        self,
        uids: Iterable[str],
        *,
        timeout: float | None = None,
        session: Session | None = None,
    ) -> DataFrame:
        unique_uids = list(dict.fromkeys(uids))
        if not unique_uids:
            self.logger.warning("No user ids supplied for OSM lookup")
            return DataFrame(columns=["id"]).set_index("id")

        timeout = timeout or self.timeout
        http = session or Session()
        owns_session = session is None

        base_url = "https://openstreetmap.org/api/0.6/users.json?users="
        fetched: list[dict[str, Any]] = []

        def fetch_batch(batch: list[str]) -> None:
            if not batch:
                return
            url = base_url + ",".join(batch)
            try:
                response = http.get(url, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
            except (RequestException, ValueError) as exc:
                if len(batch) == 1:
                    self.logger.error("Failed to fetch OSM data for uid %s: %s", batch[0], exc)
                    return
                self.logger.warning(
                    "Failed to fetch %s OSM users (%s); splitting batch", len(batch), exc
                )
                mid = max(1, len(batch) // 2)
                fetch_batch(batch[:mid])
                fetch_batch(batch[mid:])
                return

            users_payload = payload.get("users") if isinstance(payload, dict) else None
            if not isinstance(users_payload, list):
                self.logger.warning("Unexpected OSM payload for %s; skipping batch", url)
                return

            fetched.extend(users_payload)
            self.logger.info(
                "Fetched %s OSM users (running total %s)", len(users_payload), len(fetched)
            )

        def chunked(seq: list[str], size: int) -> Iterable[list[str]]:
            for start in range(0, len(seq), size):
                yield seq[start : start + size]

        try:
            for batch in chunked(unique_uids, 100):
                fetch_batch(batch)
        finally:
            if owns_session:
                http.close()

        records: list[dict[str, Any]] = []
        for entry in fetched:
            user_obj = entry.get("user") if isinstance(entry, dict) else None
            if isinstance(user_obj, dict):
                records.append(user_obj)

        if not records:
            self.logger.warning("No mapper information returned from OSM")
            return DataFrame(columns=["id"]).set_index("id")

        df = DataFrame(records)
        if "id" not in df.columns:
            self.logger.warning("OSM payload missing id column; returning raw dataframe")
            return df

        return df.drop_duplicates(subset="id").set_index("id")

    def add_uids_to_team(self, uids: Iterable[int], team: int):
        responses = []
        for uid in uids:
            try:
                responses.append(self._request("PUT", f"teams/add/{team}/{uid}"))
            except Exception:
                self.logger.exception("Failed to add uid %s to team %s", uid, team)
                raise
        return responses

    def remove_moderator_from_team(self, uid: int, team: int):
        return self._request("PUT", f"teams/{team}/removeModerator/{uid}")

    def remove_uid_from_team(self, uid: int, team: int):
        return self._request("PUT", f"teams/remove/{team}/{uid}")

    def update_team_location(self, team: int, point: dict[str, Any]):
        data = {"location": point}
        res = self._request("PUT", f"teams/{team}", json_payload=data)
        return res

    def add_attributes_to_team(self, team, attributes):
        curr_data = self.get_team(team)
        data = dumps({"hashtag": curr_data.get("hashtag"), "tags": attributes})
        res = self.session.put(self.API_URL + f"teams/{team}", data=data)
        return res

    def assign_badge(self, org_id, badge_id, user_id, assigned_at, valid_until):
        data = {"assigned_at": str(assigned_at)}
        if valid_until:
            data["valid_until"] = str(valid_until)

        res = self.session.post(
            self.API_URL + f"organizations/{org_id}/badges/{badge_id}/assign/{user_id}",
            data=dumps(data),
        )

        return res

    def get_join_link(self, team: int, session: Session | None = None):
        session = session or self.session
        return self._request("GET", f"teams/{team}/invitations", session=session)

    def create_org_team(self, body: dict[str, Any]):
        return self._request(
            "POST",
            f"organizations/{self.ORG}/teams",
            json_payload=body,
        )

    def fetch_and_cache_teams(self, cache_path: Path = TEAMS_CACHE) -> DataFrame:
        teams = self.get_all_organization_teams()
        detailed = self.fetch_team_details(teams_df=teams)
        save_teams_dataframe(detailed, cache_path)
        return detailed

    def fetch_and_cache_members(self, cache_path: Path = MEMBERS_CACHE) -> DataFrame:
        members = self.get_all_organization_members()
        enriched = self.fetch_member_details(members_df=members)
        save_members_dataframe(enriched, cache_path)
        return enriched

    def load_cached_data(
        self,
        members_path: Path = MEMBERS_CACHE,
        teams_path: Path = TEAMS_CACHE,
    ) -> tuple[DataFrame, DataFrame]:
        if not members_path.exists():
            raise FileNotFoundError(members_path)
        if not teams_path.exists():
            raise FileNotFoundError(teams_path)
        members_df = load_members_dataframe(members_path)
        teams_df = load_teams_dataframe(teams_path)
        return members_df, teams_df
