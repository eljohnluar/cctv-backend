"""College year levels, lettered sections, and per-teacher section scope.

A section name is year-scoped (``"1st Year - Section A"``) so the 5 sections that
exist in each of the 4 years stay distinguishable in reports and filters.
"""

import time
from typing import Dict, List, Optional

from database.supabase_client import get_supabase
from utils.demo_accounts import FALLBACK_USERS
from utils.logger import logger

YEAR_LEVELS = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
SECTION_LETTERS = ["A", "B", "C", "D", "E"]


def section_label(year_level: str, letter: str) -> str:
    return f"{year_level} - Section {letter}"


def normalise_years(values: Optional[List[str]]) -> List[str]:
    return [value for value in (values or []) if value in YEAR_LEVELS]


def normalise_letters(values: Optional[List[str]]) -> List[str]:
    cleaned = []
    for value in values or []:
        letter = str(value).strip().upper()[:1]
        if letter in SECTION_LETTERS and letter not in cleaned:
            cleaned.append(letter)
    return cleaned


def expand_sections(year_levels: Optional[List[str]], letters: Optional[List[str]]) -> List[str]:
    """Cartesian product of the year levels and section letters a teacher handles."""
    years = normalise_years(year_levels)
    cleaned = normalise_letters(letters)
    if not years or not cleaned:
        return []
    return [section_label(year, letter) for year in years for letter in cleaned]


def year_level_of(section: Optional[str]) -> str:
    for year in YEAR_LEVELS:
        if (section or "").startswith(year):
            return year
    return ""


_ASSIGNMENT_TTL_SECONDS = 30.0
_assignment_cache: Dict[str, tuple] = {}


def _account_assignments(username: str) -> Dict[str, List[str]]:
    cached = _assignment_cache.get(username)
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1]

    assignments = {"year_levels": [], "letters": []}
    client = get_supabase()
    if client:
        try:
            result = (
                client.table("users")
                .select("year_levels, sections")
                .eq("username", username)
                .execute()
            )
            if result.data:
                row = result.data[0]
                assignments = {
                    "year_levels": row.get("year_levels") or [],
                    "letters": row.get("sections") or [],
                }
        except Exception as error:
            logger.warning("Could not read section scope for '%s': %s", username, error)
            # Retry soon, but not on every single request.
            _assignment_cache[username] = (now + 5.0, assignments)
            return assignments
    else:
        account = FALLBACK_USERS.get(username) or {}
        assignments = {
            "year_levels": account.get("year_levels") or [],
            "letters": account.get("sections") or [],
        }

    _assignment_cache[username] = (now + _ASSIGNMENT_TTL_SECONDS, assignments)
    return assignments


def resolve_allowed_sections(claims: Optional[dict]) -> Optional[List[str]]:
    """Sections the caller may work with, or None when they are unrestricted.

    Administrators and unauthenticated internal callers (the camera worker) get
    None. A teacher is always scoped to their assignments, so an account with no
    year levels or section letters sees nothing instead of every roster.
    """
    if not claims or claims.get("role") != "teacher":
        return None

    assignments = _account_assignments(claims.get("sub", ""))
    return expand_sections(assignments["year_levels"], assignments["letters"])
