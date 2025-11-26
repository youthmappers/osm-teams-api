from __future__ import annotations

import base64
from datetime import date
import json
import logging
from typing import Any

import pandas as pd
from pandas.api import types as pd_types

try:  # pragma: no cover - optional dependency imports
    import gspread
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    GOOGLE_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when extras are missing
    GOOGLE_DEPS_AVAILABLE = False

__all__ = ["GoogleSheetsMixin"]

logger = logging.getLogger(__name__)


class GoogleSheetsMixin:
    """Mixin containing legacy Google Drive and Sheets helpers."""

    def _require_google_deps(self) -> None:
        if not GOOGLE_DEPS_AVAILABLE:
            raise RuntimeError(
                "Google integrations require the 'google' optional dependency group. "
                "Install it via `uv sync --extra google` and retry."
            )
        if getattr(self, "creds", None) is None:
            raise RuntimeError(
                "Google integrations require YGL_GOOGLE_CREDENTIALS to be configured"
            )

    # Legacy helpers below -------------------------------------------------

    def update_latest_data_on_google_drive(self) -> None:
        """Upload local members.json and chapters.json to Google Drive."""

        self._require_google_deps()
        g_credentials = service_account.Credentials.from_service_account_info(self.creds)

        drive_service = build(
            "drive",
            "v3",
            credentials=g_credentials.with_scopes(scopes=["https://www.googleapis.com/auth/drive"]),
        )

        media = MediaFileUpload("chapters.json", mimetype="application/json")
        drive_service.files().update(
            fileId=self.chapters_id,
            body={"name": "chapters.json"},
            media_body=media,
        ).execute()

        media = MediaFileUpload("members.json", mimetype="application/json")
        drive_service.files().update(
            fileId=self.members_id,
            body={"name": "members.json"},
            media_body=media,
        ).execute()

    def fetch_previous_master_list_and_conflate(self, mappers_df: pd.DataFrame) -> None:
        self._require_google_deps()
        gc = gspread.service_account_from_dict(
            json.loads(base64.b64decode(self.creds)) if isinstance(self.creds, str) else self.creds
        )
        previous_master_list_sheet = gc.open_by_key(
            "17EOKwXR8kolG_Lkz-xYH8ysLcwhDl11Zq7scHyOzakI"
        ).worksheet("Sheet1")
        previous_master_list = pd.DataFrame(previous_master_list_sheet.get_all_records()).set_index(
            "UID"
        )
        previous_master_list["source"] = "Old List"

        df = mappers_df.join(previous_master_list, rsuffix="_masterlist", how="outer")

        def _conflate_gender(row: pd.Series):
            value = self._conflate(row, "Gender", "gender", frame=df)
            return value.lower() if value is not None else None

        df.Gender = df.apply(_conflate_gender, axis=1)
        df.username = df.apply(
            lambda row: self._conflate(row, "username", "username_masterlist", frame=df), axis=1
        )
        df.Name = df.apply(
            lambda row: self._conflate(row, "Name", "Name_masterlist", frame=df), axis=1
        )
        df.Email = df.apply(
            lambda row: self._conflate(row, "Email", "email", frame=df), axis=1
        )
        df.team_id = df.apply(
            lambda row: self._conflate(row, "team_id", "chapter_id", frame=df), axis=1
        ).astype(int)
        df.source = df.apply(
            lambda row: self._conflate(row, "source", "source_masterlist", frame=df), axis=1
        )
        df.alumni_date = df.apply(
            lambda row: self._conflate(row, "alumni_date", "alumni_date_conflated", frame=df), axis=1
        )
        df["Role / Position"] = df.apply(
            lambda row: self._conflate(row, "Role / Position", "role", frame=df),
            axis=1,
        )

        df = df.drop(["graduation","phone"], axis=1, errors="ignore")

        self.mappers_df = df
        return df

    def update_latest_youthmapper_roster(self) -> None:
        self._require_google_deps()
        OUTPUT_COLUMNS = [
            "uid",
            "username",
            "Name",
            "Gender",
            "Major or Degree Concentration",
            "Graduation Date",
            "Hometown and Country",
            "Role / Position",
            "Email",
            "changeset_count",
            "account_created",
            "team_id",
            "all_teams",
            "source",
            "alumni_date",
            "Alumni",
            "Regional Ambassador",
            "Mentor / Faculty Advisor",
            "Steering Committee",
            "Chapter",
            "University",
            "City",
            "Country",
            "chapter_lon",
            "chapter_lat",
        ]

        t = (
            self.df.merge(
                self.chapters[
                    ["name", "University", "City", "Country", "chapter_lon", "chapter_lat"]
                ].rename(columns={"name": "Chapter"}),
                left_on="team_id",
                right_index=True,
            )
            .reset_index()
            .rename(columns={"index": "uid"})
        )[OUTPUT_COLUMNS]

        t.Alumni = t.Alumni.apply(
            lambda x: pd.Timestamp(x.get("assigned_at")).date() if pd.notnull(x) else ""
        )
        t["Regional Ambassador"] = t["Regional Ambassador"].apply(
            lambda x: pd.Timestamp(x.get("assigned_at")).date() if pd.notnull(x) else ""
        )
        t["Steering Committee"] = t["Steering Committee"].apply(
            lambda x: pd.Timestamp(x.get("assigned_at")).date() if pd.notnull(x) else ""
        )
        t["Mentor / Faculty Advisor"] = t["Mentor / Faculty Advisor"].apply(
            lambda x: pd.Timestamp(x.get("assigned_at")).date() if pd.notnull(x) else ""
        )

        spreadsheet = self.gc.open_by_key("1IpBO7Kuv75Ij6dNtUQ33t9PUuN7_E4DwHNOrfRY2r1Y")
        ym_sheet = spreadsheet.worksheet("YouthMappers")
        cell_list = ym_sheet.range(f"A2:{self._get_column_from_int(len(OUTPUT_COLUMNS))}{len(t)+1}")

        OUTPUT_COLUMNS_MAPPING = dict(enumerate(OUTPUT_COLUMNS))
        for cell in cell_list:
            df_row, df_col = cell.row - 2, cell.col - 1
            val = t.iloc[df_row, df_col]
            cell.value = self._format_cell_value(val, OUTPUT_COLUMNS_MAPPING.get(df_col))

        ym_sheet.update_cells(cell_list, value_input_option="USER_ENTERED")

    def update_latest_chapter_roster(self) -> None:
        self._require_google_deps()
        CHAPTER_COLUMNS = [
            "team_id",
            "name",
            "hashtag",
            "bio",
            "privacy",
            "members",
            "City",
            "Country",
            "Website or Social Media Accounts",
            "Year Established",
            "E-mail",
            "join_link",
            "chapter_lon",
            "chapter_lat",
            "location",
        ]

        t = self.chapters.reset_index().rename(columns={"index": "team_id"})[CHAPTER_COLUMNS]
        t.join_link = t.apply(
            lambda row: f"https://mapping.team/teams/{row.team_id}/invitations/{row.join_link}",
            axis=1,
        )

        spreadsheet = self.gc.open_by_key("1IpBO7Kuv75Ij6dNtUQ33t9PUuN7_E4DwHNOrfRY2r1Y")
        chapters_sheet = spreadsheet.worksheet("Chapters")
        cell_list = chapters_sheet.range(
            f"A2:{self._get_column_from_int(len(CHAPTER_COLUMNS))}{len(t)+1}"
        )

        CHAPTER_OUTPUT_COLUMNS_MAPPING = dict(enumerate(CHAPTER_COLUMNS))
        for cell in cell_list:
            df_row, df_col = cell.row - 2, cell.col - 1
            val = t.iloc[df_row, df_col]
            cell.value = self._format_cell_value(val, CHAPTER_OUTPUT_COLUMNS_MAPPING.get(df_col))

        chapters_sheet.update_cells(cell_list, value_input_option="USER_ENTERED")

    # Helper utilities -----------------------------------------------------

    @staticmethod
    def _conflate(
        row: pd.Series,
        priority_field: str,
        secondary_field: str,
        tertiary_field: str | None = None,
        *,
        frame: pd.DataFrame | None = None,
    ):
        priority_value = row.get(priority_field)
        if pd.notnull(priority_value):
            return priority_value

        target_dtype = None
        if frame is not None and priority_field in frame.columns:
            target_dtype = frame[priority_field].dtype

        secondary_value = row.get(secondary_field)
        if pd.notnull(secondary_value):
            return GoogleSheetsMixin._cast_like_priority(secondary_value, target_dtype)

        if tertiary_field:
            tertiary_value = row.get(tertiary_field)
            if pd.notnull(tertiary_value):
                return GoogleSheetsMixin._cast_like_priority(tertiary_value, target_dtype)

        return GoogleSheetsMixin._missing_like_dtype(target_dtype)

    @staticmethod
    def _cast_like_priority(value: Any, dtype) -> Any:
        if dtype is None or not pd.notnull(value):
            return value
        try:
            return pd.Series([value]).astype(dtype, errors="ignore").iloc[0]
        except Exception:  # noqa: BLE001 - best-effort casting only
            logger.debug(
                "Failed to cast value %r to dtype %s; returning original",
                value,
                dtype,
                exc_info=True,
            )
            return value

    @staticmethod
    def _missing_like_dtype(dtype):
        if dtype is None:
            return None
        if pd_types.is_datetime64_any_dtype(dtype):
            return pd.NaT
        if pd_types.is_string_dtype(dtype):
            return ""
        if pd_types.is_bool_dtype(dtype) or pd_types.is_integer_dtype(dtype) or pd_types.is_float_dtype(dtype):
            return pd.NA
        return None

    @staticmethod
    def _format_cell_value(val: Any, column: str | None):
        match column:
            case "uid":
                return int(val)
            case "username":
                return f'=HYPERLINK("https://osm.org/user/"&"{val}","{val}")'
            case "team_id":
                return f'=HYPERLINK("https://mapping.team/teams/"&{int(val)},{int(val)})'
            case "Year Born":
                return int(val) if pd.notnull(val) else ""
            case "chapter_lon" | "chapter_lat":
                return float(val) if pd.notnull(val) else ""
            case "all_teams":
                if isinstance(val, list):
                    return "; ".join(str(v) for v in val)
                return ""
            case "account_created":
                return "'" + val.strftime("%Y-%m-%d") if pd.notnull(val) else ""
            case _:
                return str(val) if pd.notnull(val) else ""

    @staticmethod
    def _get_column_from_int(number: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        idx = number % 26

        additional = ""
        if number > 26:
            additional = alphabet[int(number / 26) - 1]

        return additional + alphabet[idx - 1]
