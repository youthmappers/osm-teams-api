"""Shared response factories for tests — return dicts matching API shapes."""
from __future__ import annotations


def make_team_response(team_id: int = 1, name: str = "Test Team", **extra) -> dict:
    return {"id": team_id, "name": name, **extra}


def make_members_page(
    members: list[dict],
    page: int = 1,
    last_page: int = 1,
    total: int | None = None,
) -> dict:
    total = total if total is not None else len(members)
    return {
        "members": {
            "data": members,
            "pagination": {
                "page": page,
                "lastPage": last_page,
                "total": total,
            },
        }
    }


def make_member(osm_id: int = 1, name: str = "mapper") -> dict:
    return {"id": osm_id, "name": name}


def make_osm_users_response(uids: list[int]) -> dict:
    return {
        "users": [
            {
                "user": {
                    "id": uid,
                    "display_name": f"user_{uid}",
                    "changesets": {"count": uid * 10},
                    "traces": {"count": 0},
                    "img": {"href": f"https://example.com/{uid}.png"},
                    "account_created": "2020-01-01T00:00:00Z",
                }
            }
            for uid in uids
        ]
    }
