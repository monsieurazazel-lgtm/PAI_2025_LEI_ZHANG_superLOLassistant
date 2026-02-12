from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame, Series

# ---- Qt (PySide6) -------------------------------------------------------------
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ---- Matplotlib (Qt6 backend) ------------------------------------------------
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import rcParams
import matplotlib.pyplot as plt

# ---- Onglet InGame + hotkey --------------------------------------------------
from ingame_tab import InGameTab
import trashbase

# ---- Riot --------------------------------------------------------------------
try:
    from riotwatcher import ApiError, LolWatcher, RiotWatcher
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Installe d'abord : pip install riotwatcher pandas PySide6 matplotlib requests\n"
        + str(e)
    )


# =============================================================================
# Thème & style (QSS + couleurs)
# =============================================================================

BLUE = "#0C5D8B"
BLUE_D = "#094A70"
BLUE_L = "#2E7FAE"
ORANGE = "#F39C12"
GREY = "#F6F8FA"
BORDER = "#D9E3EC"

DARK_BG = "#1E1E1E"
DARK_CARD = "#2D2D2D"
DARK_BORDER = "#404040"
DARK_TEXT = "#E0E0E0"
DARK_GREY = "#3A3A3A"

APP_QSS_LIGHT = f"""
* {{ font-family:'Segoe UI','Inter','Roboto','Noto',sans-serif; font-size:12.5px; }}
QMainWindow {{ background:white; }}
QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:8px; padding:6px; }}
QTabBar::tab {{
  background:{GREY}; color:{BLUE}; border:1px solid {BORDER}; border-bottom:none;
  padding:6px 12px; margin-right:4px; border-top-left-radius:8px; border-top-right-radius:8px;
}}
QTabBar::tab:selected {{ background:white; color:{BLUE}; border:1px solid {BORDER}; border-bottom:2px solid white; }}
QGroupBox {{ border:1px solid {BORDER}; border-radius:8px; margin-top:18px; background:white; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; top:-2px; padding:0 6px; color:{BLUE}; font-weight:600; background:white; }}
QPushButton {{ background:{BLUE}; color:white; border:none; border-radius:8px; padding:7px 12px; }}
QPushButton:hover {{ background:{BLUE_D}; }}
QPushButton#primaryBtn {{ background:{ORANGE}; }}
QPushButton#primaryBtn:hover {{ background:#d1800a; }}
QLineEdit, QSpinBox, QComboBox {{ background:white; border:1px solid {BORDER}; border-radius:6px; padding:6px 8px; color:black; }}
QTableWidget {{ gridline-color:{BORDER}; background:white; alternate-background-color:{GREY}; border:1px solid {BORDER}; border-radius:8px; }}
QHeaderView::section {{ background:{BLUE}; color:white; padding:6px; border:none; }}
QLabel {{ color: black; }}
#banner {{ background: white; border:1px solid {BORDER}; border-radius:10px; padding:4px 8px; }}
#brandTitle {{ color:{BLUE}; font-size:18px; font-weight:800; }}
#brandSubtitle {{ color:{ORANGE}; font-size:12px; font-weight:700; margin-left:4px; }}
#smallTip {{ color:#5c6b77; font-size:12px; }}
"""

APP_QSS_DARK = f"""
* {{ font-family:'Segoe UI','Inter','Roboto','Noto',sans-serif; font-size:12.5px; color:{DARK_TEXT}; }}
QMainWindow {{ background:{DARK_BG}; }}
QTabWidget::pane {{ border:1px solid {DARK_BORDER}; border-radius:8px; padding:6px; background:{DARK_BG}; }}
QTabBar::tab {{
  background:{DARK_GREY}; color:{BLUE_L}; border:1px solid {DARK_BORDER}; border-bottom:none;
  padding:6px 12px; margin-right:4px; border-top-left-radius:8px; border-top-right-radius:8px;
}}
QTabBar::tab:selected {{ background:{DARK_CARD}; color:{BLUE_L}; border:1px solid {DARK_BORDER}; border-bottom:2px solid {DARK_CARD}; }}
QGroupBox {{ border:1px solid {DARK_BORDER}; border-radius:8px; margin-top:18px; background:{DARK_CARD}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; top:-2px; padding:0 6px; color:{BLUE_L}; font-weight:600; background:{DARK_CARD}; }}
QPushButton {{ background:{BLUE}; color:white; border:none; border-radius:8px; padding:7px 12px; }}
QPushButton:hover {{ background:{BLUE_L}; }}
QPushButton#primaryBtn {{ background:{ORANGE}; }}
QPushButton#primaryBtn:hover {{ background:#e8a81c; }}
QLineEdit, QSpinBox, QComboBox {{ background:{DARK_GREY}; border:1px solid {DARK_BORDER}; border-radius:6px; padding:6px 8px; color:{DARK_TEXT}; }}
QTableWidget {{ gridline-color:{DARK_BORDER}; background:{DARK_CARD}; alternate-background-color:{DARK_GREY}; border:1px solid {DARK_BORDER}; border-radius:8px; color:{DARK_TEXT}; }}
QTableWidget::item {{ color:{DARK_TEXT}; }}
QHeaderView::section {{ background:{BLUE}; color:white; padding:6px; border:none; }}
QLabel {{ color:{DARK_TEXT}; }}
QListWidget {{ background:{DARK_GREY}; border:1px solid {DARK_BORDER}; color:{DARK_TEXT}; }}
QListWidget::item:selected {{ background:{BLUE}; }}
#banner {{ background:{DARK_CARD}; border:1px solid {DARK_BORDER}; border-radius:10px; padding:4px 8px; }}
#brandTitle {{ color:{BLUE_L}; font-size:18px; font-weight:800; }}
#brandSubtitle {{ color:{ORANGE}; font-size:12px; font-weight:700; margin-left:4px; }}
#smallTip {{ color:#888888; font-size:12px; }}
"""

# Matplotlib defaults
rcParams["axes.facecolor"] = "white"
rcParams["figure.facecolor"] = "white"
rcParams["font.size"] = 11
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11


# =============================================================================
# Constantes / utilitaires généraux
# =============================================================================

ROLES: list[str] = ["top", "jungle", "mid", "bot", "sup"]

ROLE_MAP: dict[str, str] = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "UTILITY": "sup",
}

PLATFORM_GROUPS: dict[str, list[str]] = {
    "europe": ["euw1", "eune1", "tr1", "ru"],
    "americas": ["na1", "br1", "la1", "la2"],
    "asia": ["kr", "jp1"],
    "sea": ["oc1"],
}

SLEEP_PER_CALL = 1.25
BACKOFF_429 = 3.0


def _df_is_empty(df: Optional[DataFrame]) -> bool:
    """Check if a DataFrame is None or empty.

    Args:
        df: DataFrame to test.

    Returns:
        True if df is None or df.empty, otherwise False.
    """
    return df is None or bool(df.empty)


def _series_is_empty(s: Optional[Series]) -> bool:
    """Check if a Series is None or empty.

    Args:
        s: Series to test.

    Returns:
        True if s is None or s.empty, otherwise False.
    """
    return s is None or bool(s.empty)


def _mask_any(mask: Any) -> bool:
    """Safely evaluate whether a boolean mask contains any True."""
    # Si c'est déjà un booléen simple, on le renvoie
    if isinstance(mask, bool):
        return mask

    # Pour les objets Pandas (Series/DataFrame)
    if isinstance(mask, (Series, DataFrame)):
        # .fillna(False) pour gérer les NaN, puis .any()
        # On appelle .any() une deuxième fois pour les DataFrame
        val = mask.fillna(False).any()
        while isinstance(val, (Series, DataFrame, pd.Index)):
            val = val.any()
        return bool(val)

    # Pour NumPy ou autres itérables (cas général sécurisé)
    try:
        return bool(np.any(mask)) if hasattr(mask, "__iter__") else bool(mask)
    except Exception:
        return bool(mask)


def sleep_brief() -> None:
    """Sleep briefly to reduce Riot API spam.

    This function is used after successful API calls to keep a minimal delay
    between requests and lower the chance of hitting rate limits.

    Returns:
        None.
    """
    time.sleep(SLEEP_PER_CALL)


def safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a Riot API function with a minimal retry policy.

    The function retries on HTTP 429 (rate limit), and exits the program
    on authentication/authorization errors (401/403).

    Args:
        fn: Callable corresponding to a RiotWatcher/LolWatcher endpoint method.
        *args: Positional arguments passed to the endpoint.
        **kwargs: Keyword arguments passed to the endpoint.

    Returns:
        The API response returned by `fn`.

    Raises:
        SystemExit: If API key is invalid/expired (401/403).
        ApiError: For non-429 Riot API errors.
    """
    while True:
        try:
            res = fn(*args, **kwargs)
            sleep_brief()
            return res
        except ApiError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                time.sleep(BACKOFF_429)
                continue
            if code in (401, 403):
                raise SystemExit("Clé API invalide/expirée (401/403).")
            raise


# =============================================================================
# Lecture CSV & normalisation
# =============================================================================


def _as_series(df: DataFrame, col: Optional[str]) -> Series:
    """Return a DataFrame column as a Series, or a typed empty Series if missing.

    Args:
        df: Source DataFrame.
        col: Column name to fetch.

    Returns:
        `df[col]` if the column exists, otherwise an empty Series with dtype "object".
    """
    if col is None or col not in df.columns:
        return pd.Series(dtype="object")
    s = df[col]
    return cast(Series, s)


def _is_boolish_series(s: Series) -> float:
    """Estimate how much a Series looks like a boolean column.

    The heuristic tests for common boolean-like tokens:
    true/false, 1/0, t/f, yes/no.

    Args:
        s: Input Series.

    Returns:
        A score in [0, 1] = fraction of values that look boolean-like.
    """
    vals = cast(Series, s.astype(str).str.strip().str.lower())
    return float(
        cast(
            Series, vals.isin(["true", "false", "1", "0", "t", "f", "yes", "no"])
        ).mean()
    )


def _looks_like_role_series(s: Series) -> float:
    """Estimate how much a Series looks like a role column.

    Args:
        s: Input Series.

    Returns:
        A score in [0, 1] = fraction of values in {'top','jungle','mid','bot','sup'}.
    """
    vals = cast(Series, s.astype(str).str.strip().str.lower())
    return float(cast(Series, vals.isin(ROLES)).mean())


def _maybe_matchid_series(s: Series) -> float:
    """Estimate how much a Series looks like a Riot matchId.

    Typical match IDs look like: EUW1_1234567890 (region prefix + underscore + id).

    Args:
        s: Input Series.

    Returns:
        A score in [0, 1] = fraction of strings matching the matchId regex.
    """
    m = cast(Series, s.astype(str).str.strip().str.match(r"^[A-Z]{2,4}\d?_.+"))
    m2 = cast(Series, m.fillna(False))
    return float(m2.mean())


def _maybe_puuid_series(s: Series) -> float:
    """Estimate how much a Series looks like a PUUID.

    PUUIDs are long identifiers (often 30+ characters).

    Args:
        s: Input Series.

    Returns:
        A score in [0, 1] = fraction of strings with length >= 30.
    """
    ok = cast(Series, s.astype(str).str.len() >= 30)
    ok2 = cast(Series, ok.fillna(False))
    return float(ok2.mean())


def _guess_kda_columns(
    df: DataFrame, excluded: set[str]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Guess the kills/deaths/assists columns using numeric heuristics.

    The heuristic:
    - keeps numeric-looking columns,
    - discards low-variance columns (nunique <= 6),
    - discards implausible ranges (min < 0 or max > 60),
    - then uses mean ordering: deaths ~ smallest mean, kills/assists ~ largest means.

    Args:
        df: Source DataFrame.
        excluded: Column names to ignore (already matched as role/win/champion).

    Returns:
        A tuple (kills_col, deaths_col, assists_col). Each entry can be None if not found.
    """
    candidates: list[tuple[str, float]] = []

    for c in df.columns:
        cc = str(c)
        if cc in excluded:
            continue

        s = cast(Series, df[cc])
        sn = cast(Series, pd.to_numeric(s, errors="coerce"))
        sn_no_na = cast(Series, sn.dropna())

        if bool(sn_no_na.empty):
            continue

        if int(sn_no_na.nunique()) <= 6:
            continue

        mn = float(sn_no_na.min())
        mx = float(sn_no_na.max())
        if mn < 0 or mx > 60:
            continue

        candidates.append((cc, float(sn_no_na.mean())))

    if len(candidates) < 3:
        return (None, None, None)

    candidates.sort(key=lambda t: t[1])
    deaths = candidates[0][0]
    top2 = sorted(candidates[-2:], key=lambda t: t[1], reverse=True)
    kills, assists = top2[0][0], top2[1][0]
    return (kills, deaths, assists)


def read_csv_flexible(path: Path) -> DataFrame:
    """Read a CSV file by trying multiple separators and fallback strategies.

    The function first tries common separators (comma, semicolon, tab, pipe).
    If parsing yields too few columns, it retries without header and creates
    generic column names.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame (best effort).
    """
    seps = [",", ";", "\t", "|"]

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 6:
                return df
        except Exception:
            continue

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, header=None)
            if df.shape[1] >= 6:
                df.columns = [f"C{i}" for i in range(df.shape[1])]
                return df
        except Exception:
            continue

    return pd.read_csv(path)


def normalize_dataframe_columns(df: DataFrame) -> tuple[DataFrame, dict[str, str]]:
    """Normalize a participants-like DataFrame into standard columns.

    Standard columns:
    - role (lowercase among ROLES)
    - teamWin (boolean)
    - championName (string)
    - kills, deaths, assists (int)
    Optional if available:
    - matchId (string)
    - teamId (nullable int)
    - winnerTeamId (nullable int)

    The function uses heuristics to detect columns when names differ.

    Args:
        df: Raw DataFrame loaded from participants.csv (or similar).

    Returns:
        A tuple (df_std, mapping) where:
        - df_std: normalized DataFrame
        - mapping: dict standard_name -> original_column_name

    Raises:
        ValueError: If required columns cannot be inferred.
    """
    needed = ["role", "teamWin", "championName", "kills", "deaths", "assists"]

    if all(c in df.columns for c in needed):
        out = df.copy()
        out["role"] = cast(Series, out["role"]).astype(str).str.lower()
        out["teamWin"] = (
            cast(Series, out["teamWin"])
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes", "y"])
        )
        for c in ["kills", "deaths", "assists"]:
            sn = cast(Series, pd.to_numeric(cast(Series, out[c]), errors="coerce"))
            out[c] = cast(Series, sn.fillna(0)).astype(int)
        return out, {c: c for c in needed}

    cols: list[str] = [str(c) for c in df.columns]

    # --- role_col (avoid max(..., key=...) to keep Pyright calm)
    role_col: Optional[str] = None
    best = -1.0
    for c in cols:
        sc = _looks_like_role_series(cast(Series, df[c]))
        if sc > best:
            best = sc
            role_col = c
    if best < 0.5:
        role_col = None

    # --- win_col
    win_col: Optional[str] = None
    best = -1.0
    for c in cols:
        sc = _is_boolish_series(cast(Series, df[c]))
        if sc > best:
            best = sc
            win_col = c
    if best < 0.5:
        win_col = None

    champ_col: Optional[str] = None
    best_score = -1.0
    for c in cols:
        if c in {role_col, win_col}:
            continue
        s = cast(Series, df[c])

        if _maybe_matchid_series(s) > 0.5 or _maybe_puuid_series(s) > 0.5:
            continue

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            uniq = float(s.astype(str).nunique(dropna=True))
            med_len = float(cast(Series, s.astype(str).str.len()).median(skipna=True))
            score = uniq + (20.0 - abs(10.0 - (med_len or 10.0)))
            if score > best_score:
                champ_col = c
                best_score = score

    excluded = set(filter(None, [role_col, win_col, champ_col]))
    kills_col, deaths_col, assists_col = _guess_kda_columns(df, excluded)

    matchId_col: Optional[str] = None
    teamId_col: Optional[str] = None
    winnerTeamId_col: Optional[str] = None

    for c in cols:
        if _maybe_matchid_series(cast(Series, df[c])) > 0.5:
            matchId_col = c
            break

    for c in cols:
        sn = cast(Series, pd.to_numeric(cast(Series, df[c]), errors="coerce"))
        if (
            float(cast(Series, sn.notna()).mean()) > 0.9
            and float(cast(Series, sn.dropna().isin([100, 200])).mean()) > 0.8
        ):
            teamId_col = c
            break

    if teamId_col is not None:
        for c in cols:
            if c == teamId_col:
                continue
            sn = cast(Series, pd.to_numeric(cast(Series, df[c]), errors="coerce"))
            if (
                float(cast(Series, sn.notna()).mean()) > 0.6
                and float(cast(Series, sn.dropna().isin([100, 200])).mean()) > 0.6
            ):
                winnerTeamId_col = c
                break

    mapping: dict[str, str] = {}
    if role_col:
        mapping["role"] = role_col
    if win_col:
        mapping["teamWin"] = win_col
    if champ_col:
        mapping["championName"] = champ_col
    if kills_col:
        mapping["kills"] = kills_col
    if deaths_col:
        mapping["deaths"] = deaths_col
    if assists_col:
        mapping["assists"] = assists_col
    if matchId_col:
        mapping["matchId"] = matchId_col
    if teamId_col:
        mapping["teamId"] = teamId_col
    if winnerTeamId_col:
        mapping["winnerTeamId"] = winnerTeamId_col

    missing = [k for k in needed if k not in mapping]
    if missing:
        raise ValueError("Colonnes manquantes: " + ", ".join(missing))

    kills_s = cast(
        Series, pd.to_numeric(_as_series(df, mapping["kills"]), errors="coerce")
    )
    deaths_s = cast(
        Series, pd.to_numeric(_as_series(df, mapping["deaths"]), errors="coerce")
    )
    assists_s = cast(
        Series, pd.to_numeric(_as_series(df, mapping["assists"]), errors="coerce")
    )

    out = pd.DataFrame(
        {
            "role": _as_series(df, mapping["role"]).astype(str).str.lower(),
            "teamWin": _as_series(df, mapping["teamWin"])
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes", "y"]),
            "championName": _as_series(df, mapping["championName"]).astype(str),
            "kills": cast(Series, kills_s.fillna(0)).astype(int),
            "deaths": cast(Series, deaths_s.fillna(0)).astype(int),
            "assists": cast(Series, assists_s.fillna(0)).astype(int),
        }
    )

    if "matchId" in mapping:
        out["matchId"] = _as_series(df, mapping["matchId"]).astype(str)

    if "teamId" in mapping:
        t = cast(
            Series, pd.to_numeric(_as_series(df, mapping["teamId"]), errors="coerce")
        )
        out["teamId"] = cast(Series, t.astype("Int64"))

    if "winnerTeamId" in mapping:
        w = cast(
            Series,
            pd.to_numeric(_as_series(df, mapping["winnerTeamId"]), errors="coerce"),
        )
        out["winnerTeamId"] = cast(Series, w.astype("Int64"))

    return out, mapping


# =============================================================================
# Champion tags (DDragon)
# =============================================================================


def _ddragon_versions() -> list[str]:
    """Fetch the list of available Data Dragon versions.

    Returns:
        A list of version strings. The first element is usually the latest.
    """
    r = requests.get(
        "https://ddragon.leagueoflegends.com/api/versions.json", timeout=10
    )
    r.raise_for_status()
    return cast(list[str], r.json())


def _ddragon_champions_json(v: str, loc: str = "en_US") -> dict[str, Any]:
    """Fetch the champions JSON for a given Data Dragon version.

    Args:
        v: Data Dragon version (e.g., "14.1.1").
        loc: Locale string for champion names (default "en_US").

    Returns:
        The parsed JSON dictionary.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    r = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{v}/data/{loc}/champion.json",
        timeout=15,
    )
    r.raise_for_status()
    return cast(dict[str, Any], r.json())


def fetch_champion_primary_tags_online() -> dict[str, str]:
    """Download primary tags (classes) for each champion from Data Dragon.

    Returns:
        A dict mapping championName -> primaryTag (e.g., "Ahri" -> "Mage").
    """
    v = _ddragon_versions()[0]
    data = _ddragon_champions_json(v, "en_US")
    out: dict[str, str] = {}
    champs = cast(dict[str, Any], data.get("data") or {})
    for name, info in champs.items():
        tags = cast(list[str], info.get("tags") or [])
        out[str(name)] = str(tags[0]) if tags else "Other"
    return out


def load_champion_primary_tags() -> dict[str, str]:
    """Load champion primary tags from disk cache, otherwise fetch online.

    Cache strategy:
    - If `data/champion_tags.json` exists, try to parse it.
    - Otherwise, download from Data Dragon and save it.
    - On any failure, return an empty dict.

    Returns:
        A dict mapping championName -> primaryTag.
    """
    cache = Path("data") / "champion_tags.json"

    try:
        if cache.exists():
            with open(cache, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj:
                champs = cast(dict[str, Any], obj["data"])
                return {
                    str(k): (cast(list[str], v.get("tags") or ["Other"])[0])
                    for k, v in champs.items()
                }
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass

    try:
        tags = fetch_champion_primary_tags_online()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
        return tags
    except Exception:
        return {}


PRIMARY_TAGS = load_champion_primary_tags()


def champion_primary_tag(name: str) -> str:
    """Get the primary tag (class) for a champion.

    Args:
        name: Champion name (as used in Data Dragon / Riot API).

    Returns:
        The primary class tag (e.g., "Mage"), or "Other" if unknown.
    """
    return PRIMARY_TAGS.get(str(name), "Other")


# =============================================================================
# Statistiques utilisateur (profil)
# =============================================================================


def compute_player_overview(dfu: DataFrame) -> dict[str, Any]:
    """Compute global user stats (games, wins, winrate, KDA).

    Args:
        dfu: User participants DataFrame, must contain:
            - teamWin (bool)
            - kills, deaths, assists (int)

    Returns:
        A dict with keys:
            - games (int)
            - wins (int)
            - wr (float): winrate percentage
            - kda (float): (kills + assists) / max(1, deaths)
    """
    if bool(dfu.empty):
        return {"games": 0, "wins": 0, "wr": 0.0, "kda": 0.0}

    games = int(len(dfu))
    wins = int(cast(Series, dfu["teamWin"]).sum())
    wr = wins / games * 100.0

    kills = int(cast(Series, dfu["kills"]).sum())
    assists = int(cast(Series, dfu["assists"]).sum())
    deaths = int(cast(Series, dfu["deaths"]).sum())
    kda = (kills + assists) / max(1, deaths)

    return {"games": games, "wins": wins, "wr": wr, "kda": kda}


def compute_player_by_role(dfu: DataFrame) -> DataFrame:
    """Aggregate user stats by role.

    Args:
        dfu: User participants DataFrame.

    Returns:
        A DataFrame with columns:
            role, games, wins, wr%, kda, K, D, A
        Sorted by winrate descending.
    """
    if bool(dfu.empty):
        return pd.DataFrame(
            columns=pd.Index(["role", "games", "wins", "wr%", "kda", "K", "D", "A"])
        )

    g: DataFrame = dfu.copy()
    g["kda_per_game"] = (g["kills"] + g["assists"]) / g["deaths"].clip(lower=1)

    agg = cast(
        DataFrame,
        g.groupby("role", as_index=False).agg(
            games=("teamWin", "size"),
            wins=("teamWin", "sum"),
            K=("kills", "sum"),
            D=("deaths", "sum"),
            A=("assists", "sum"),
            kda=("kda_per_game", "mean"),
        ),
    )
    agg["wr%"] = agg["wins"] / agg["games"] * 100.0

    out = cast(
        DataFrame, agg.loc[:, ["role", "games", "wins", "wr%", "kda", "K", "D", "A"]]
    )
    out = cast(DataFrame, out.sort_values(by="wr%", ascending=False))
    return out


def compute_player_by_champ(dfu: DataFrame) -> DataFrame:
    """Aggregate user stats by champion.

    Args:
        dfu: User participants DataFrame.

    Returns:
        A DataFrame with columns:
            champion, games, wins, wr%, kda, K, D, A
    """
    if bool(dfu.empty):
        return pd.DataFrame(
            columns=pd.Index(["champion", "games", "wins", "wr%", "kda", "K", "D", "A"])
        )

    g: DataFrame = dfu.copy()
    g["kda_per_game"] = (g["kills"] + g["assists"]) / g["deaths"].clip(lower=1)

    agg = cast(
        DataFrame,
        g.groupby("championName", as_index=False).agg(
            games=("teamWin", "size"),
            wins=("teamWin", "sum"),
            K=("kills", "sum"),
            D=("deaths", "sum"),
            A=("assists", "sum"),
            kda=("kda_per_game", "mean"),
        ),
    )
    agg["wr%"] = agg["wins"] / agg["games"] * 100.0
    agg = cast(DataFrame, agg.rename(columns={"championName": "champion"}))

    out = cast(
        DataFrame,
        agg.loc[:, ["champion", "games", "wins", "wr%", "kda", "K", "D", "A"]],
    )
    return out


# =============================================================================
# META (global)
# =============================================================================


class MetaTab(QWidget):
    """META tab: tier lists (S/A/B/C) per role + matchup table on champion click."""

    def __init__(self) -> None:
        """Initialize the META tab UI and internal state."""
        super().__init__()
        self.df_std: Optional[DataFrame] = None

        self.lineCsvPath = QLineEdit()
        default_csv = Path("data_db") / "participants.csv"
        self.lineCsvPath.setText(
            str(default_csv.resolve()) if default_csv.exists() else ""
        )

        self.btnBrowse = QPushButton("Parcourir…")
        self.btnBrowse.clicked.connect(self.on_browse)

        self.spinMinGames = QSpinBox()
        self.spinMinGames.setRange(1, 100000)
        self.spinMinGames.setValue(30)

        self.comboMetric: QComboBox = QComboBox()
        self.comboMetric.addItems(["Winrate", "KDA"])

        self.btnRefresh = QPushButton("Actualiser")
        self.btnRefresh.clicked.connect(self.refresh)

        self.lblStatus = QLabel("Prêt.")
        self.lblStatus.setObjectName("smallTip")

        self.tblMatchups = QTableWidget(0, 6)
        self.tblMatchups.setHorizontalHeaderLabels(
            ["Opponent", "Games", "W-L", "Winrate %", "KDA", "K/D/A"]
        )
        self.tblMatchups.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tblMatchups.horizontalHeader().setStretchLastSection(True)
        self.tblMatchups.setAlternatingRowColors(True)

        left = QVBoxLayout()

        gbData = QGroupBox("Données")
        v = QVBoxLayout()
        r1 = QHBoxLayout()
        r1.addWidget(self.lineCsvPath)
        r1.addWidget(self.btnBrowse)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Min games"))
        r2.addWidget(self.spinMinGames)
        v.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Métrique"))
        r3.addWidget(self.comboMetric)
        v.addLayout(r3)

        v.addWidget(self.btnRefresh)
        v.addWidget(self.lblStatus)
        gbData.setLayout(v)
        left.addWidget(gbData)

        gbMU = QGroupBox("Matchups (clic un champion)")
        muV = QVBoxLayout()
        muV.addWidget(self.tblMatchups)
        gbMU.setLayout(muV)
        left.addWidget(gbMU)

        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)

        self.tabsRoles = QTabWidget()
        self.role_lists: dict[str, dict[str, QListWidget]] = {}
        self.role_info: dict[str, QLabel] = {}

        for role in ROLES:
            tab = QWidget()
            grid = QGridLayout()

            gbS, lvS = self._make_list_box("S")
            gbA, lvA = self._make_list_box("A")
            gbB, lvB = self._make_list_box("B")
            gbC, lvC = self._make_list_box("C")

            for lv in (lvS, lvA, lvB, lvC):
                lv.itemClicked.connect(
                    lambda item, r=role: self.on_champion_clicked(r, item)
                )

            grid.addWidget(gbS, 0, 0)
            grid.addWidget(gbA, 0, 1)
            grid.addWidget(gbB, 1, 0)
            grid.addWidget(gbC, 1, 1)

            info = QLabel(f"{role.capitalize()} — 0 champions")
            info.setObjectName("smallTip")
            grid.addWidget(info, 2, 0, 1, 2)

            tab.setLayout(grid)
            self.tabsRoles.addTab(tab, role.capitalize())
            self.role_lists[role] = {"S": lvS, "A": lvA, "B": lvB, "C": lvC}
            self.role_info[role] = info

        main = QHBoxLayout()
        main.addWidget(lw, 1)
        main.addWidget(self.tabsRoles, 3)
        self.setLayout(main)

    def _make_list_box(self, title: str) -> tuple[QGroupBox, QListWidget]:
        """Create a titled list box for tier display.

        Args:
            title: Tier label (e.g., "S", "A", "B", "C").

        Returns:
            A tuple (groupbox, listwidget).
        """
        gb = QGroupBox(title)
        lv = QListWidget()
        vv = QVBoxLayout()
        vv.addWidget(lv)
        gb.setLayout(vv)
        return gb, lv

    def on_browse(self) -> None:
        """Open a file dialog to select a participants CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir participants.csv", "", "CSV (*.csv);;All (*.*)"
        )
        if path:
            self.lineCsvPath.setText(path)

    def refresh(self) -> None:
        """Reload CSV data, compute per-role tiers, and refresh the UI."""
        csv_path = self.lineCsvPath.text().strip() or "data_db/participants.csv"

        try:
            df_raw = read_csv_flexible(Path(csv_path))
            df_std, _mapping = normalize_dataframe_columns(df_raw)
            self.df_std = df_std
        except Exception as e:
            QMessageBox.critical(
                self, "Erreur CSV", f"Impossible de lire '{csv_path}':\n{e}"
            )
            return

        min_games = int(self.spinMinGames.value())
        metric = self.comboMetric.currentText().strip()

        dfx: DataFrame = cast(DataFrame, self.df_std.copy())
        dfx["kda_per_game"] = (dfx["kills"] + dfx["assists"]) / dfx["deaths"].clip(
            lower=1
        )

        per_role: dict[str, DataFrame] = {}
        for role in ROLES:
            sub = cast(DataFrame, dfx.loc[dfx["role"] == role])
            if bool(sub.empty):
                per_role[role] = pd.DataFrame()
                continue

            grp = cast(
                DataFrame,
                sub.groupby("championName", as_index=False).agg(
                    games=("teamWin", "size"),
                    wins=("teamWin", "sum"),
                    K=("kills", "sum"),
                    D=("deaths", "sum"),
                    A=("assists", "sum"),
                    kda_mean=("kda_per_game", "mean"),
                ),
            )
            grp["winrate_percent"] = grp["wins"] / grp["games"] * 100.0
            per_role[role] = grp

        for role in ROLES:
            for t in ("S", "A", "B", "C"):
                self.role_lists[role][t].clear()

            data0 = per_role.get(role)
            if data0 is None or bool(data0.empty):
                self.role_info[role].setText(f"{role.capitalize()} — 0 champions")
                continue

            data = cast(DataFrame, data0.loc[data0["games"] >= min_games].copy())
            if bool(data.empty):
                self.role_info[role].setText(
                    f"{role.capitalize()} — 0 champions (filtrés)"
                )
                continue

            if metric == "Winrate":
                data["Tier"] = cast(Series, data["winrate_percent"]).apply(
                    self._tier_from_winrate
                )
                data = cast(
                    DataFrame,
                    data.sort_values(
                        by=["Tier", "winrate_percent", "games"],
                        ascending=[True, False, False],
                    ),
                )
                for _, row in data.iterrows():
                    tier = str(row["Tier"])
                    self.role_lists[role][tier].addItem(
                        f"{row['championName']} — {row['winrate_percent']:.1f}% ({int(row['games'])})"
                    )
            else:
                data["Tier"] = cast(Series, data["kda_mean"]).apply(self._tier_from_kda)
                data = cast(
                    DataFrame,
                    data.sort_values(
                        by=["Tier", "kda_mean", "games"], ascending=[True, False, False]
                    ),
                )
                for _, row in data.iterrows():
                    tier = str(row["Tier"])
                    self.role_lists[role][tier].addItem(
                        f"{row['championName']} — KDA {row['kda_mean']:.2f} ({int(row['games'])})"
                    )

            counts = {
                t: int(self.role_lists[role][t].count()) for t in ("S", "A", "B", "C")
            }
            total = int(sum(counts.values()))
            self.role_info[role].setText(
                f"{role.capitalize()} — {total} champions  |  S:{counts['S']}  A:{counts['A']}  B:{counts['B']}  C:{counts['C']}"
            )

    def _tier_from_winrate(self, wr: float) -> str:
        """Convert a winrate into a tier.

        Args:
            wr: Winrate percentage (0..100).

        Returns:
            "S", "A", "B", or "C".
        """
        if wr >= 54:
            return "S"
        if wr >= 51:
            return "A"
        if wr >= 48:
            return "B"
        return "C"

    def _tier_from_kda(self, kda: float) -> str:
        """Convert a KDA value into a tier.

        Args:
            kda: KDA metric (kills+assists)/deaths.

        Returns:
            "S", "A", "B", or "C".
        """
        if kda >= 3.5:
            return "S"
        if kda >= 3.0:
            return "A"
        if kda >= 2.5:
            return "B"
        return "C"

    def on_champion_clicked(self, role: str, item: Any) -> None:
        """Compute and display matchup stats for a clicked champion.

        Requires matchId/teamId columns to exist in the normalized CSV.

        Args:
            role: Role currently displayed (top/jungle/mid/bot/sup).
            item: QListWidgetItem-like object containing the champion label.

        Returns:
            None.
        """
        if self.df_std is None:
            return

        champ = str(item.text()).split("—", 1)[0].strip()
        df = self.df_std

        if "matchId" not in df.columns or "teamId" not in df.columns:
            QMessageBox.information(
                self, "Matchups", "matchId/teamId absents dans le CSV."
            )
            return

        sub = cast(
            DataFrame,
            df.loc[(df["role"] == role) & (df["championName"] == champ)].copy(),
        )
        if bool(sub.empty):
            return

        df2: DataFrame = df.copy()
        df2["kda_per_game"] = (df2["kills"] + df2["assists"]) / df2["deaths"].clip(
            lower=1
        )
        idx = df2.set_index(["matchId", "role"]).sort_index()

        rec: list[tuple[str, bool, int, int, int, float]] = []
        for _, row in sub.iterrows():
            mid = row.get("matchId")
            my_team = row.get("teamId")
            my_team = row.get("teamId")
            is_mid_na = bool(pd.isna(mid))
            is_team_na = bool(pd.isna(my_team))

            if is_mid_na or is_team_na:
                continue

            try:
                group = idx.loc[(mid, role)]
                if isinstance(group, Series):
                    group = group.to_frame().T
                group_df = cast(DataFrame, group)
            except KeyError:
                continue

            opp_rows = cast(DataFrame, group_df.loc[group_df["teamId"] != my_team])
            if bool(opp_rows.empty):
                continue

            opp = opp_rows.iloc[0]
            oc = str(opp["championName"])

            win = bool(row["teamWin"])
            k = int(row["kills"])
            d = int(row["deaths"])
            a = int(row["assists"])
            kda = (k + a) / max(1, d)
            rec.append((oc, win, k, d, a, kda))

        if not rec:
            return

        mu = pd.DataFrame(
            rec,
            columns=pd.Index(["opponent", "win", "K", "D", "A", "kda"]),
        )

        agg = cast(
            DataFrame,
            mu.groupby("opponent", as_index=False).agg(
                games=("win", "size"),
                wins=("win", "sum"),
                K=("K", "sum"),
                D=("D", "sum"),
                A=("A", "sum"),
                kda=("kda", "mean"),
            ),
        )
        agg["losses"] = agg["games"] - agg["wins"]
        agg["wr_percent"] = agg["wins"] / agg["games"] * 100.0

        agg = cast(
            DataFrame,
            agg.loc[
                :,
                [
                    "opponent",
                    "games",
                    "wins",
                    "losses",
                    "wr_percent",
                    "kda",
                    "K",
                    "D",
                    "A",
                ],
            ],
        )
        agg = cast(
            DataFrame,
            agg.sort_values(by=["wr_percent", "games"], ascending=[False, False]),
        )

        self._fill_matchups_table(agg)

    def _fill_matchups_table(self, agg: DataFrame) -> None:
        """Fill the matchups QTableWidget from an aggregated DataFrame.

        Args:
            agg: Aggregated matchups DataFrame containing:
                opponent, games, wins, losses, wr_percent, kda, K, D, A

        Returns:
            None.
        """
        self.tblMatchups.setRowCount(0)
        self.tblMatchups.setRowCount(int(len(agg)))

        for r, (_, row) in enumerate(agg.iterrows()):
            kdastr = f"{int(row['K'])}/{int(row['D'])}/{int(row['A'])}"
            vals = [
                str(row["opponent"]),
                str(int(row["games"])),
                f"{int(row['wins'])}-{int(row['losses'])}",
                f"{float(row['wr_percent']):.1f}",
                f"{float(row['kda']):.2f}",
                kdastr,
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (1, 3, 4):
                    it.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.tblMatchups.setItem(r, c, it)

        self.tblMatchups.resizeColumnsToContents()
        self.tblMatchups.horizontalHeader().setStretchLastSection(True)


# =============================================================================
# Overlay de mini-camembert
# =============================================================================


class PieOverlay(QWidget):
    """Floating overlay widget displaying a small pie chart for drill-down."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the overlay window and its Matplotlib canvas.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setWindowOpacity(0.95)
        self.setFixedSize(340, 250)

        self.fig = plt.figure(figsize=(3.0, 2.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        self.btnClose = QPushButton("✕")
        self.btnClose.setFixedSize(22, 22)
        self.btnClose.clicked.connect(self.close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 8)
        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self.btnClose, 0, Qt.AlignmentFlag.AlignRight)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

    def show_pie(
        self,
        title: str,
        labels: list[str],
        sizes: list[int],
        pos: Optional[QPoint] = None,
    ) -> None:
        """Render and show a pie chart in the overlay.

        Args:
            title: Title displayed above the pie chart.
            labels: Slice labels.
            sizes: Slice sizes (must match labels length).
            pos: Optional screen position for the overlay. If None, it uses the current cursor position.

        Returns:
            None.
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        colors = [BLUE, BLUE_L, ORANGE, "#7FB3D5", "#F7C87A", "#B2D7EF", "#FFD08A"]
        _ = ax.pie(
            sizes,
            labels=labels,
            colors=colors[: len(labels)],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        )

        ax.set_title(title, color=BLUE, fontsize=11)

        if pos is None:
            gp = QCursor.pos()
            pos = QPoint(gp.x() + 14, gp.y() + 14)

        self.move(pos)
        self.canvas.draw()
        self.show()


# =============================================================================
# MON PROFIL
# =============================================================================


class ProfileTab(QWidget):
    """Profile tab: Riot collection / personal CSV load / stats tables / pie charts."""

    def __init__(self) -> None:
        """Initialize the PROFILE tab UI and internal state."""
        super().__init__()

        self.dfu: Optional[DataFrame] = None
        self.overlay = PieOverlay(self)
        self._pie_conn: Optional[int] = None

        self.lineApiKey = QLineEdit(os.getenv("RIOT_API_KEY") or "")
        self.lineApiKey.setPlaceholderText("RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

        self.lineRiotID = QLineEdit()
        self.lineRiotID.setPlaceholderText("gameName#tagLine (ex: ztheo17#EUW)")

        self.comboRegion = QComboBox()
        self.comboRegion.addItems(["europe", "americas", "asia", "sea"])

        self.comboPlatform = QComboBox()
        self.comboPlatform.addItems(
            [
                "euw1",
                "na1",
                "kr",
                "eune1",
                "br1",
                "jp1",
                "la1",
                "la2",
                "oc1",
                "tr1",
                "ru",
            ]
        )

        self.spinTarget = QSpinBox()
        self.spinTarget.setRange(10, 1000)
        self.spinTarget.setValue(50)

        self.comboQueue = QComboBox()
        self.comboQueue.addItems(["420 (SoloQ)", "440 (Flex)", "0 (Toutes)"])

        self.btnBuild = QPushButton("Construire ma DB (test: 1 fois)")
        self.btnBuild.setObjectName("primaryBtn")
        self.btnBuild.clicked.connect(self.on_build)

        self.btnLoadCSV = QPushButton("Charger CSV perso…")
        self.btnLoadCSV.clicked.connect(self.on_load_csv)

        self.lblInfo = QLabel("")

        self.comboSortChamp = QComboBox()
        self.comboSortChamp.addItems(
            ["Winrate", "Games", "KDA", "Kills", "Deaths", "Assists"]
        )
        self.comboSortChamp.currentIndexChanged.connect(self.update_user_views)

        self.lblSummary = QLabel("—")
        self.lblSummary.setObjectName("smallTip")

        summaryBox = QGroupBox("Winrate")
        sv = QVBoxLayout()
        sv.addWidget(self.lblSummary)
        summaryBox.setLayout(sv)

        self.tblRoles = QTableWidget(0, 8)
        self.tblRoles.setHorizontalHeaderLabels(
            ["Role", "Games", "Wins", "WR %", "KDA", "K", "D", "A"]
        )
        self.tblRoles.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tblRoles.itemSelectionChanged.connect(self.on_role_selected)
        self.tblRoles.setAlternatingRowColors(True)

        self.tblChamps = QTableWidget(0, 8)
        self.tblChamps.setHorizontalHeaderLabels(
            ["Champion", "Games", "Wins", "WR %", "KDA", "K", "D", "A"]
        )
        self.tblChamps.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tblChamps.setAlternatingRowColors(True)

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        controls = QGridLayout()
        controls.addWidget(QLabel("Riot API Key"), 0, 0)
        controls.addWidget(self.lineApiKey, 0, 1, 1, 3)
        controls.addWidget(QLabel("Riot ID"), 1, 0)
        controls.addWidget(self.lineRiotID, 1, 1, 1, 3)
        controls.addWidget(QLabel("Region (account/match)"), 2, 0)
        controls.addWidget(self.comboRegion, 2, 1)
        controls.addWidget(QLabel("Platform (league/summoner)"), 2, 2)
        controls.addWidget(self.comboPlatform, 2, 3)
        controls.addWidget(QLabel("Target matches"), 3, 0)
        controls.addWidget(self.spinTarget, 3, 1)
        controls.addWidget(QLabel("Queue"), 3, 2)
        controls.addWidget(self.comboQueue, 3, 3)
        controls.addWidget(self.btnBuild, 4, 0, 1, 2)
        controls.addWidget(self.btnLoadCSV, 4, 2, 1, 2)
        controls.addWidget(self.lblInfo, 5, 0, 1, 4)

        topBox = QGroupBox("Construire / Charger DB perso")
        tv = QVBoxLayout()
        tv.addLayout(controls)
        topBox.setLayout(tv)

        leftCol = QVBoxLayout()
        rolesBox = QGroupBox("Par rôle (sélectionne une ligne pour voir le camembert)")
        vr = QVBoxLayout()
        vr.addWidget(self.tblRoles)
        rolesBox.setLayout(vr)

        champsBox = QGroupBox("Par champion")
        vc = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Trier par"))
        row1.addWidget(self.comboSortChamp)
        row1.addStretch(1)
        vc.addLayout(row1)
        vc.addWidget(self.tblChamps)
        champsBox.setLayout(vc)

        leftCol.addWidget(rolesBox)
        leftCol.addWidget(champsBox)

        rightCol = QVBoxLayout()
        chartBox = QGroupBox("Types joués (camembert)")
        cv = QVBoxLayout()
        cv.addWidget(self.canvas)
        chartBox.setLayout(cv)

        rightCol.addWidget(summaryBox)
        rightCol.addWidget(chartBox, 1)

        middle = QHBoxLayout()
        middle.addLayout(leftCol, 1)
        middle.addLayout(rightCol, 1)

        main = QVBoxLayout()
        main.addWidget(topBox)
        main.addLayout(middle)

        self.setLayout(main)

    def _get_api_key(self) -> Optional[str]:
        """Retrieve the API key from input field or environment.

        Returns:
            The API key string, or None if missing.
        """
        return self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY")

    def on_build(self) -> None:
        """Collect user matches via Riot API and build a personal dataset.

        This method:
        - validates Riot ID + API key,
        - collects up to `target` matches,
        - writes a CSV under data_user/,
        - updates the UI with computed stats.

        Returns:
            None.
        """
        riot_id = re.sub(r"\s+", "", self.lineRiotID.text())
        if not riot_id:
            QMessageBox.warning(self, "Riot ID", "Saisis un Riot ID: gameName#tagLine")
            return

        api_key = self._get_api_key()
        if not api_key:
            QMessageBox.critical(self, "API", "Renseigne ta Riot API Key.")
            return

        region = self.comboRegion.currentText().strip().lower()
        platform = self.comboPlatform.currentText().strip().lower()
        target = int(self.spinTarget.value())

        qtxt = self.comboQueue.currentText().strip()
        queue = (
            420 if qtxt.startswith("420") else (440 if qtxt.startswith("440") else 0)
        )

        self.lblInfo.setText("Collecte en cours…")
        QApplication.processEvents()

        try:
            dfu = collect_user_participants(
                api_key,
                region,
                platform,
                riot_id,
                target_matches=target,
                queue_id=(queue or None),
            )
        except Exception as e:
            QMessageBox.critical(self, "Collecte", f"Échec collecte:\n{e}")
            self.lblInfo.setText("Échec.")
            return

        if bool(dfu.empty):
            QMessageBox.information(self, "Collecte", "Aucun match récupéré.")
            self.lblInfo.setText("Aucun match.")
            return

        outdir = Path("data_user")
        outdir.mkdir(parents=True, exist_ok=True)
        outcsv = outdir / f"participants_{riot_id.replace('#', '_')}.csv"
        dfu.to_csv(outcsv, index=False)

        self.lblInfo.setText(f"Chargé: {outcsv.resolve()}")
        self.dfu = dfu
        self.update_user_views()

    def on_load_csv(self) -> None:
        """Load a user CSV (already in the normalized schema) from disk.

        Required columns:
            role, teamWin, championName, kills, deaths, assists

        Returns:
            None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger CSV perso", "", "CSV (*.csv);;All (*.*)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Lecture impossible:\n{e}")
            return

        needed = {"role", "teamWin", "championName", "kills", "deaths", "assists"}
        if not needed.issubset(set(df.columns)):
            QMessageBox.critical(
                self,
                "CSV",
                "Colonnes requises: role, teamWin, championName, kills, deaths, assists",
            )
            return

        df["role"] = cast(Series, df["role"]).astype(str).str.lower()
        df["teamWin"] = (
            cast(Series, df["teamWin"])
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes", "y"])
        )
        for c in ["kills", "deaths", "assists"]:
            sn = cast(Series, pd.to_numeric(cast(Series, df[c]), errors="coerce"))
            df[c] = cast(Series, sn.fillna(0)).astype(int)

        self.dfu = df
        self.update_user_views()

    def role_pie(self, role: str) -> None:
        """Draw the 'types played' pie chart for a given role.

        The pie groups champions by their primary Data Dragon tag.

        Args:
            role: One of ROLES.

        Returns:
            None.
        """
        self.fig.clear()

        if _df_is_empty(self.dfu):
            self.canvas.draw()
            return

        assert self.dfu is not None
        sub = cast(DataFrame, self.dfu.loc[self.dfu["role"] == role])
        if bool(sub.empty):
            self.canvas.draw()
            return

        counts: dict[str, int] = {}
        vc = cast(Series, sub["championName"]).value_counts()
        for champ, n in vc.items():
            t = champion_primary_tag(str(champ))
            counts[t] = counts.get(t, 0) + int(n)

        total = int(sum(counts.values()))
        if total == 0:
            self.canvas.draw()
            return

        labels = list(counts.keys())
        sizes = [int(v) for v in counts.values()]

        ax = self.fig.add_subplot(111)
        ax.clear()

        colors = [BLUE, BLUE_L, ORANGE, "#7FB3D5", "#F7C87A", "#B2D7EF", "#FFD08A"]
        res = ax.pie(
            sizes,
            labels=labels,
            colors=colors[: len(labels)],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        )
        wedges = res[0]

        ax.set_title(f"Répartition des types — {role.capitalize()}", color=BLUE)

        for w in wedges:
            w.set_picker(True)

        if self._pie_conn is not None:
            self.canvas.mpl_disconnect(self._pie_conn)

        self._pie_conn = self.canvas.mpl_connect(
            "pick_event",
            lambda ev, r=role, lab=labels, sz=sizes, sdf=sub: self.on_pie_pick(
                ev, r, lab, sz, sdf
            ),
        )

        self.canvas.draw()

    def on_pie_pick(
        self,
        event: Any,
        role: str,
        labels: list[str],
        sizes: list[int],
        sub_df: DataFrame,
    ) -> None:
        """Handle click on a pie slice to show drill-down overlay.

        Args:
            event: Matplotlib pick_event payload.
            role: Role associated with the chart.
            labels: Tag labels used in the main pie.
            sizes: Tag sizes used in the main pie (kept for stable signature).
            sub_df: Subset DataFrame filtered to the selected role.

        Returns:
            None.
        """
        _ = sizes  # keep the signature stable
        wedge = event.artist
        ax = event.canvas.figure.gca()

        wedges = [p for p in ax.patches if isinstance(p, type(wedge))]
        if wedge not in wedges:
            return

        idx = wedges.index(wedge)
        tag = labels[idx]

        vc = cast(Series, sub_df["championName"]).value_counts()
        items: list[tuple[str, int]] = []
        for name, n in vc.items():
            if champion_primary_tag(str(name)) == tag:
                items.append((str(name), int(n)))

        if not items:
            return

        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:8]
        rest = int(sum(n for _, n in items[8:]))
        if rest > 0:
            top.append(("Autres", rest))

        clabels = [k for k, _ in top]
        csizes = [v for _, v in top]
        self.overlay.show_pie(f"{tag} — {role}", clabels, csizes, None)

    def on_role_selected(self) -> None:
        """Update the pie chart when a role row is selected in the roles table."""
        items = self.tblRoles.selectedItems()
        if not items:
            return
        row = items[0].row()
        role = cast(QTableWidgetItem, self.tblRoles.item(row, 0)).text()
        self.role_pie(role)

    def update_user_views(self) -> None:
        """Refresh summary label, tables, and default selected role chart."""
        if _df_is_empty(self.dfu):
            self.lblSummary.setText("—")
            self.tblRoles.setRowCount(0)
            self.tblChamps.setRowCount(0)
            self.fig.clear()
            self.canvas.draw()
            return

        assert self.dfu is not None
        ov = compute_player_overview(self.dfu)
        self.lblSummary.setText(
            f"Games: {ov['games']} | Wins: {ov['wins']} | WR: {ov['wr']:.1f}% | KDA: {ov['kda']:.2f}"
        )

        self._fill_roles_table(compute_player_by_role(self.dfu))
        self._fill_champs_table(compute_player_by_champ(self.dfu))

        if self.tblRoles.rowCount() > 0:
            self.tblRoles.selectRow(0)
            self.on_role_selected()

    def _fill_roles_table(self, rdf: DataFrame) -> None:
        """Fill the roles QTableWidget.

        Args:
            rdf: DataFrame from compute_player_by_role().

        Returns:
            None.
        """
        self.tblRoles.setRowCount(0)
        self.tblRoles.setRowCount(int(len(rdf)))

        for r, (_, row) in enumerate(rdf.iterrows()):
            vals = [
                str(row["role"]),
                int(row["games"]),
                int(row["wins"]),
                f"{float(row['wr%']):.1f}",
                f"{float(row['kda']):.2f}",
                int(row["K"]),
                int(row["D"]),
                int(row["A"]),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c in (1, 2, 3, 4, 5, 6, 7):
                    it.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.tblRoles.setItem(r, c, it)

        self.tblRoles.resizeColumnsToContents()
        self.tblRoles.horizontalHeader().setStretchLastSection(True)

    def _fill_champs_table(self, cdf: DataFrame) -> None:
        """Fill the champions QTableWidget with sorting.

        Args:
            cdf: DataFrame from compute_player_by_champ().

        Returns:
            None.
        """
        sk = self.comboSortChamp.currentText().strip()

        if sk == "Winrate":
            cdf = cast(
                DataFrame,
                cdf.sort_values(by=["wr%", "games"], ascending=[False, False]),
            )
        elif sk == "Games":
            cdf = cast(DataFrame, cdf.sort_values(by="games", ascending=False))
        elif sk == "KDA":
            cdf = cast(DataFrame, cdf.sort_values(by="kda", ascending=False))
        elif sk == "Kills":
            cdf = cast(DataFrame, cdf.sort_values(by="K", ascending=False))
        elif sk == "Deaths":
            cdf = cast(DataFrame, cdf.sort_values(by="D", ascending=True))
        elif sk == "Assists":
            cdf = cast(DataFrame, cdf.sort_values(by="A", ascending=False))

        self.tblChamps.setRowCount(0)
        self.tblChamps.setRowCount(int(len(cdf)))

        for r, (_, row) in enumerate(cdf.iterrows()):
            vals = [
                str(row["champion"]),
                int(row["games"]),
                int(row["wins"]),
                f"{float(row['wr%']):.1f}",
                f"{float(row['kda']):.2f}",
                int(row["K"]),
                int(row["D"]),
                int(row["A"]),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c in (1, 2, 3, 4, 5, 6, 7):
                    it.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.tblChamps.setItem(r, c, it)

        self.tblChamps.resizeColumnsToContents()
        self.tblChamps.horizontalHeader().setStretchLastSection(True)


# =============================================================================
# Collecte perso depuis Riot (profil)
# =============================================================================


def normalize_riot_id(s: str) -> Optional[tuple[str, str]]:
    """Normalize and validate a Riot ID string.

    Expected input format:
        gameName#tagLine

    Args:
        s: Raw Riot ID string.

    Returns:
        (gameName, TAGLINE) if valid, otherwise None.
    """
    if not s:
        return None
    s2 = re.sub(r"\s+", "", s.strip())
    if "#" not in s2:
        return None
    g, t = s2.split("#", 1)
    if not g or not t:
        return None
    return g, t.upper()


def riotid_to_puuid(rw: RiotWatcher, region: str, rid: str) -> Optional[str]:
    """Convert a Riot ID (gameName#tagLine) to a PUUID.

    Args:
        rw: RiotWatcher instance (account endpoints).
        region: Region group for account endpoints (e.g., "europe").
        rid: Riot ID string.

    Returns:
        The puuid string if found, otherwise None.
    """
    tup = normalize_riot_id(rid)
    if not tup:
        return None

    game, tag = tup
    acc = safe_call(rw.account.by_riot_id, region, game, tag)
    if isinstance(acc, dict):
        return cast(Optional[str], acc.get("puuid"))
    return None


def collect_user_participants(
    api_key: str,
    region: str,
    platform: str,
    riot_id: str,
    target_matches: int = 50,
    queue_id: Optional[int] = 420,
) -> DataFrame:
    """Collect a user's matches from Riot and return a participants-like DataFrame.

    Notes:
        - `platform` is kept in the signature for future extensions (league/summoner endpoints),
          but this function currently uses account/match endpoints.

    Args:
        api_key: Riot API key (RGAPI-...).
        region: Match/account region group (e.g., "europe").
        platform: Platform routing (e.g., "euw1"). Currently unused.
        riot_id: Riot ID in the form gameName#tagLine.
        target_matches: Number of matches to fetch (max 100 per API call).
        queue_id: Optional queue filter (420 SoloQ, 440 Flex). If None/0, no filter.

    Returns:
        DataFrame with columns:
            matchId, teamId, role, teamWin, championName,
            kills, deaths, assists, kda_ratio, opponentChampion

    Raises:
        ValueError: If Riot ID cannot be resolved to a PUUID.
    """
    _ = platform

    rw = RiotWatcher(api_key)
    lol = LolWatcher(api_key)

    puuid = riotid_to_puuid(rw, region, riot_id)
    if not puuid:
        raise ValueError("Riot ID introuvable")

    kw: dict[str, Any] = {}
    if queue_id:
        kw.update(queue=queue_id, type="ranked")

    mlist = safe_call(
        lol.match.matchlist_by_puuid,
        region,
        puuid,
        count=min(100, target_matches),
        **kw,
    )
    if not mlist:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for mid in cast(list[str], mlist)[:target_matches]:
        try:
            match = safe_call(lol.match.by_id, region, mid)
        except ApiError:
            continue

        if not isinstance(match, dict):
            continue

        info = cast(dict[str, Any], match.get("info", {}) or {})
        parts = cast(list[dict[str, Any]], info.get("participants", []) or [])

        me: Optional[dict[str, Any]] = None
        for p in parts:
            if p.get("puuid") == puuid:
                me = p
                break
        if not me:
            continue

        role = ROLE_MAP.get(str(me.get("teamPosition") or "").upper())
        if not role:
            continue

        my_team = me.get("teamId")

        opp_name: Optional[str] = None
        for p in parts:
            if (
                ROLE_MAP.get(str(p.get("teamPosition") or "").upper()) == role
                and p.get("teamId") != my_team
            ):
                opp_name = cast(Optional[str], p.get("championName"))
                break

        k = int(me.get("kills", 0))
        d = int(me.get("deaths", 0))
        a = int(me.get("assists", 0))
        kda = (k + a) / (d if d > 0 else 1)

        meta = cast(dict[str, Any], match.get("metadata", {}) or {})
        rows.append(
            {
                "matchId": meta.get("matchId"),
                "teamId": my_team,
                "role": role,
                "teamWin": bool(me.get("win")),
                "championName": me.get("championName"),
                "kills": k,
                "deaths": d,
                "assists": a,
                "kda_ratio": float(f"{kda:.3f}"),
                "opponentChampion": opp_name,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Fenêtre principale
# =============================================================================


class MainWindow(QMainWindow):
    """Main window: banner + tabs (META / PROFILE / IN GAME) + theme + hotkey."""

    def __init__(self) -> None:
        """Initialize the main window and its tabs."""
        super().__init__()

        self.setWindowTitle("SuperLOL Assistant")
        self.setMinimumSize(1100, 720)

        self.is_dark_mode = False
        self.hotkey_char: Optional[str] = None

        self.ingame_tab = InGameTab()

        banner = self._build_banner()

        self.tabs = QTabWidget()
        self.tabs.addTab(MetaTab(), "META (global)")
        self.tabs.addTab(ProfileTab(), "MON PROFIL")
        self.tabs.addTab(self.ingame_tab, "IN GAME")

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(banner)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _build_banner(self) -> QWidget:
        """Build the banner widget (logo + title + theme toggle + hotkey button).

        Returns:
            The banner QWidget.
        """
        banner = QWidget()
        banner.setObjectName("banner")

        bl = QHBoxLayout(banner)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(8)

        logo_path = self._find_logo()
        if logo_path:
            pm = QPixmap(logo_path).scaledToHeight(
                54, Qt.TransformationMode.SmoothTransformation
            )
            pic = QLabel()
            pic.setPixmap(pm)
            bl.addWidget(
                pic, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.setWindowIcon(QIcon(logo_path))

        titleWrap = QHBoxLayout()
        lab1 = QLabel("SuperLOL")
        lab1.setObjectName("brandTitle")
        lab2 = QLabel("Assistant")
        lab2.setObjectName("brandSubtitle")
        titleWrap.addWidget(lab1)
        titleWrap.addWidget(lab2)
        titleWrap.addStretch(1)

        self.btnTheme = QPushButton("🌙 dark~")
        self.btnTheme.setFixedWidth(100)
        self.btnTheme.clicked.connect(self.toggle_theme)

        self.btnHotkey = QPushButton("设定发送键")
        self.btnHotkey.setFixedWidth(100)
        self.btnHotkey.clicked.connect(self.configure_hotkey)

        titleWrap.addWidget(
            self.btnTheme,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        titleWrap.addWidget(
            self.btnHotkey,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        bl.addLayout(titleWrap, 1)
        return banner

    def _find_logo(self) -> Optional[str]:
        """Find a logo file among common names in the current directory.

        Returns:
            The logo path as string if found, otherwise None.
        """
        for name in ("superlol_logo.png", "logo.png", "superlol.png"):
            p = Path(name)
            if p.exists():
                return str(p)
        return None

    def toggle_theme(self) -> None:
        """Toggle between light and dark QSS themes."""
        self.is_dark_mode = not self.is_dark_mode
        app = cast(Optional[QApplication], QApplication.instance())
        if app is None:
            return

        if self.is_dark_mode:
            app.setStyleSheet(APP_QSS_DARK)
            self.btnTheme.setText("☀️ dayday~")
        else:
            app.setStyleSheet(APP_QSS_LIGHT)
            self.btnTheme.setText("🌙 dark~")

    def configure_hotkey(self) -> None:
        """Configure a single-character hotkey to send in-game summary via trashbase."""
        from PySide6.QtWidgets import QInputDialog

        key, ok = QInputDialog.getText(
            self, "设定热键", "输入一个键（如 1 或 - ）用来发送当前局内数据："
        )
        if not ok or not key:
            return

        key = key.strip()
        if len(key) != 1:
            QMessageBox.warning(self, "热键", "只能设置单个字符键。")
            return

        self.hotkey_char = key

        def on_trigger() -> str:
            """Hotkey callback used by trashbase.

            Returns:
                The current in-game summary text (may be empty).
            """
            txt = self.ingame_tab.get_live_summary_text()
            return txt or ""

        trashbase.start_hotkey_listener(self.hotkey_char, on_trigger)
        QMessageBox.information(
            self, "热键已设定", f"按下 '{self.hotkey_char}' 将发送当前局内扫描摘要。"
        )


# =============================================================================
# main
# =============================================================================


def main() -> None:
    """Application entry point: create the Qt app and show the main window."""
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS_LIGHT)

    w = MainWindow()
    w.resize(1280, 760)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
