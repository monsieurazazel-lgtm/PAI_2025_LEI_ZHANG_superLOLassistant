#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, re, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ingame_tab import InGameTab
import trashbase

import pandas as pd
import requests

# ---- Qt / Matplotlib ---------------------------------------------------------
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QIcon, QPixmap, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QListWidget,
    QTabWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- Riot --------------------------------------------------------------------
try:
    from riotwatcher import LolWatcher, RiotWatcher, ApiError
except Exception as e:
    raise SystemExit(
        "Installe d'abord : pip install riotwatcher pandas PySide6 matplotlib requests\n"
        + str(e)
    )

# =============================================================================
# Thème & style
# =============================================================================

BLUE = "#0C5D8B"
BLUE_D = "#094A70"
BLUE_L = "#2E7FAE"
ORANGE = "#F39C12"
GREY = "#F6F8FA"
BORDER = "#D9E3EC"

# 黑夜模式颜色
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

rcParams["axes.facecolor"] = "white"
rcParams["figure.facecolor"] = "white"
rcParams["font.size"] = 11
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11

# =============================================================================
# Constantes / utilitaires
# =============================================================================

ROLES = ["top", "jungle", "mid", "bot", "sup"]
ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "UTILITY": "sup",
}

PLATFORM_GROUPS: Dict[str, List[str]] = {
    "europe": ["euw1", "eune1", "tr1", "ru"],
    "americas": ["na1", "br1", "la1", "la2"],
    "asia": ["kr", "jp1"],
    "sea": ["oc1"],
}

SLEEP_PER_CALL = 1.25
BACKOFF_429 = 3.0


def sleep_brief():
    time.sleep(SLEEP_PER_CALL)


def safe_call(fn, *args, **kwargs):
    """Retry basique + respect 429"""
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


def _is_boolish_series(s: pd.Series) -> float:
    vals = s.astype(str).str.strip().str.lower()
    return vals.isin({"true", "false", "1", "0"}).mean()


def _looks_like_role_series(s: pd.Series) -> float:
    vals = s.astype(str).str.strip().str.lower()
    return vals.isin(ROLES).mean()


def _maybe_matchid_series(s: pd.Series) -> float:
    return (
        s.astype(str).str.strip().str.match(r"^[A-Z]{2,4}\d?_.+").fillna(False).mean()
    )


def _maybe_puuid_series(s: pd.Series) -> float:
    return (s.astype(str).str.len() >= 30).fillna(False).mean()


def _is_intish(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except:
        return False


def _guess_kda_columns(
    df: pd.DataFrame, excluded: set[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cand = []
    for c in df.columns:
        if c in excluded:
            continue
        if not _is_intish(df[c]):
            continue
        sn = pd.to_numeric(df[c], errors="coerce")
        if sn.dropna().nunique() <= 6:
            continue
        mn, mx = sn.min(skipna=True), sn.max(skipna=True)
        if mn >= 0 and mx <= 60:
            cand.append((c, sn.mean(skipna=True)))
    if len(cand) < 3:
        return (None, None, None)
    cand.sort(key=lambda t: t[1])
    deaths = cand[0][0]
    k, a = [c for c, _ in sorted(cand[-2:], key=lambda t: t[1], reverse=True)]
    return (k, deaths, a)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 6:
                return df
        except:
            pass
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None)
            if df.shape[1] >= 6:
                df.columns = [f"C{i}" for i in range(df.shape[1])]
                return df
        except:
            pass
    return pd.read_csv(path)


def normalize_dataframe_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    needed = ["role", "teamWin", "championName", "kills", "deaths", "assists"]
    if all(c in df.columns for c in needed):
        out = df.copy()
        out["role"] = out["role"].astype(str).str.lower()
        out["teamWin"] = (
            out["teamWin"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        )
        for c in ["kills", "deaths", "assists"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        return out, {c: c for c in needed}

    cols = list(df.columns)
    role_col = max(cols, key=lambda c: _looks_like_role_series(df[c]), default=None)
    if _looks_like_role_series(df.get(role_col, pd.Series([]))) < 0.5:
        role_col = None

    win_col = max(cols, key=lambda c: _is_boolish_series(df[c]), default=None)
    if _is_boolish_series(df.get(win_col, pd.Series([]))) < 0.5:
        win_col = None

    champ_col = None
    best = -1
    for c in cols:
        if c in {role_col, win_col}:
            continue
        s = df[c]
        if _maybe_matchid_series(s) > 0.5 or _maybe_puuid_series(s) > 0.5:
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            uniq = s.astype(str).nunique(dropna=True)
            med = s.astype(str).str.len().median(skipna=True)
            score = uniq + (20 - abs(10 - (med or 10)))
            if score > best:
                champ_col = c
                best = score

    excluded = set(filter(None, [role_col, win_col, champ_col]))
    kills_col, deaths_col, assists_col = _guess_kda_columns(df, excluded)

    matchId_col = teamId_col = winnerTeamId_col = None
    for c in cols:
        if _maybe_matchid_series(df[c]) > 0.5:
            matchId_col = c
            break
    for c in cols:
        sn = pd.to_numeric(df[c], errors="coerce")
        if sn.notna().mean() > 0.9 and sn.dropna().isin([100, 200]).mean() > 0.8:
            teamId_col = c
            break
    if teamId_col:
        for c in cols:
            if c == teamId_col:
                continue
            sn = pd.to_numeric(df[c], errors="coerce")
            if sn.notna().mean() > 0.6 and sn.dropna().isin([100, 200]).mean() > 0.6:
                winnerTeamId_col = c
                break

    mapping = {}
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

    out = pd.DataFrame(
        {
            "role": df[mapping["role"]].astype(str).str.lower(),
            "teamWin": df[mapping["teamWin"]]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes", "y"]),
            "championName": df[mapping["championName"]].astype(str),
            "kills": pd.to_numeric(df[mapping["kills"]], errors="coerce")
            .fillna(0)
            .astype(int),
            "deaths": pd.to_numeric(df[mapping["deaths"]], errors="coerce")
            .fillna(0)
            .astype(int),
            "assists": pd.to_numeric(df[mapping["assists"]], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )
    if "matchId" in mapping:
        out["matchId"] = df[mapping["matchId"]].astype(str)
    if "teamId" in mapping:
        out["teamId"] = pd.to_numeric(df[mapping["teamId"]], errors="coerce").astype(
            "Int64"
        )
    if "winnerTeamId" in mapping:
        out["winnerTeamId"] = pd.to_numeric(
            df[mapping["winnerTeamId"]], errors="coerce"
        ).astype("Int64")
    return out, mapping


# =============================================================================
# Champion tags (DDragon)
# =============================================================================


def _ddragon_versions() -> list[str]:
    r = requests.get(
        "https://ddragon.leagueoflegends.com/api/versions.json", timeout=10
    )
    r.raise_for_status()
    return r.json()


def _ddragon_champions_json(v: str, loc: str = "en_US") -> dict:
    r = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{v}/data/{loc}/champion.json",
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def fetch_champion_primary_tags_online() -> dict[str, str]:
    v = _ddragon_versions()[0]
    data = _ddragon_champions_json(v, "en_US")
    out = {}
    for name, info in (data.get("data") or {}).items():
        tags = info.get("tags") or []
        out[str(name)] = str(tags[0]) if tags else "Other"
    return out


def load_champion_primary_tags() -> dict[str, str]:
    cache = Path("data") / "champion_tags.json"
    try:
        if cache.exists():
            with open(cache, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj:
                return {
                    str(k): (v.get("tags", ["Other"])[0] if v.get("tags") else "Other")
                    for k, v in obj["data"].items()
                }
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
    except:
        pass
    try:
        tags = fetch_champion_primary_tags_online()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
        return tags
    except:
        return {}


PRIMARY_TAGS = load_champion_primary_tags()


def champion_primary_tag(name: str) -> str:
    return PRIMARY_TAGS.get(str(name), "Other")


# =============================================================================
# Statistiques utilisateur (profil)
# =============================================================================


def compute_player_overview(dfu: pd.DataFrame) -> Dict[str, Any]:
    if dfu.empty:
        return {"games": 0, "wins": 0, "wr": 0.0, "kda": 0.0}
    games = len(dfu)
    wins = int(dfu["teamWin"].sum())
    wr = wins / games * 100.0
    kda = (dfu["kills"].sum() + dfu["assists"].sum()) / max(1, int(dfu["deaths"].sum()))
    return {"games": games, "wins": wins, "wr": wr, "kda": kda}


def compute_player_by_role(dfu: pd.DataFrame) -> pd.DataFrame:
    if dfu.empty:
        return pd.DataFrame(
            columns=["role", "games", "wins", "wr%", "kda", "K", "D", "A"]
        )
    g = dfu.copy()
    g["kda_per_game"] = (g["kills"] + g["assists"]) / g["deaths"].clip(lower=1)
    agg = g.groupby("role", as_index=False).agg(
        games=("teamWin", "size"),
        wins=("teamWin", "sum"),
        K=("kills", "sum"),
        D=("deaths", "sum"),
        A=("assists", "sum"),
        kda=("kda_per_game", "mean"),
    )
    agg["wr%"] = agg["wins"] / agg["games"] * 100.0
    return agg[["role", "games", "wins", "wr%", "kda", "K", "D", "A"]].sort_values(
        "wr%", ascending=False
    )


def compute_player_by_champ(dfu: pd.DataFrame) -> pd.DataFrame:
    if dfu.empty:
        return pd.DataFrame(
            columns=["champion", "games", "wins", "wr%", "kda", "K", "D", "A"]
        )
    g = dfu.copy()
    g["kda_per_game"] = (g["kills"] + g["assists"]) / g["deaths"].clip(lower=1)
    agg = g.groupby("championName", as_index=False).agg(
        games=("teamWin", "size"),
        wins=("teamWin", "sum"),
        K=("kills", "sum"),
        D=("deaths", "sum"),
        A=("assists", "sum"),
        kda=("kda_per_game", "mean"),
    )
    agg["wr%"] = agg["wins"] / agg["games"] * 100.0
    agg = agg.rename(columns={"championName": "champion"})
    return agg[["champion", "games", "wins", "wr%", "kda", "K", "D", "A"]]


# =============================================================================
# META (global)
# =============================================================================


class MetaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df_std: Optional[pd.DataFrame] = None

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
        self.comboMetric = QComboBox()
        self.comboMetric.addItems(["Winrate", "KDA"])
        self.btnRefresh = QPushButton("Actualiser")
        self.btnRefresh.clicked.connect(self.refresh)
        self.lblStatus = QLabel("Prêt.")
        self.lblStatus.setObjectName("smallTip")

        self.tblMatchups = QTableWidget(0, 6)
        self.tblMatchups.setHorizontalHeaderLabels(
            ["Opponent", "Games", "W-L", "Winrate %", "KDA", "K/D/A"]
        )
        self.tblMatchups.setEditTriggers(QTableWidget.NoEditTriggers)
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
        self.role_lists: Dict[str, Dict[str, QListWidget]] = {}
        self.role_info: Dict[str, QLabel] = {}
        for role in ROLES:
            tab = QWidget()
            grid = QGridLayout()

            def mk(title):
                gb = QGroupBox(title)
                lv = QListWidget()
                vv = QVBoxLayout()
                vv.addWidget(lv)
                gb.setLayout(vv)
                return gb, lv

            gbS, lvS = mk("S")
            gbA, lvA = mk("A")
            gbB, lvB = mk("B")
            gbC, lvC = mk("C")
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

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir participants.csv", "", "CSV (*.csv);;All (*.*)"
        )
        if path:
            self.lineCsvPath.setText(path)

    def refresh(self):
        csv_path = self.lineCsvPath.text().strip() or "data_db/participants.csv"
        try:
            df_raw = read_csv_flexible(Path(csv_path))
            df, mapping = normalize_dataframe_columns(df_raw)
            self.df_std = df
        except Exception as e:
            QMessageBox.critical(
                self, "Erreur CSV", f"Impossible de lire '{csv_path}':\n{e}"
            )
            return
        # self.lblStatus.setText(
        #     "Mapping: " + ", ".join([f"{k}←{v}" for k, v in mapping.items()])
        # )

        min_games = int(self.spinMinGames.value())
        metric = self.comboMetric.currentText()
        per_role = {}
        dfx = self.df_std.copy()
        dfx["kda_per_game"] = (dfx["kills"] + dfx["assists"]) / dfx["deaths"].clip(
            lower=1
        )

        for role in ROLES:
            sub = dfx[dfx["role"] == role]
            if sub.empty:
                per_role[role] = pd.DataFrame()
                continue
            grp = sub.groupby("championName", as_index=False).agg(
                games=("teamWin", "size"),
                wins=("teamWin", "sum"),
                K=("kills", "sum"),
                D=("deaths", "sum"),
                A=("assists", "sum"),
                kda_mean=("kda_per_game", "mean"),
            )
            grp["winrate_percent"] = grp["wins"] / grp["games"] * 100.0
            per_role[role] = grp

        for role in ROLES:
            for t in ("S", "A", "B", "C"):
                self.role_lists[role][t].clear()
            data = per_role.get(role)
            if data is None or data.empty:
                self.role_info[role].setText(f"{role.capitalize()} — 0 champions")
                continue
            data = data[data["games"] >= min_games].copy()
            if data.empty:
                self.role_info[role].setText(
                    f"{role.capitalize()} — 0 champions (filtrés)"
                )
                continue

            if metric == "Winrate":

                def to_tier(wr):
                    return (
                        "S"
                        if wr >= 54
                        else ("A" if wr >= 51 else ("B" if wr >= 48 else "C"))
                    )

                data["Tier"] = data["winrate_percent"].apply(to_tier)
                data = data.sort_values(
                    ["Tier", "winrate_percent", "games"], ascending=[True, False, False]
                )
                for _, row in data.iterrows():
                    self.role_lists[role][row["Tier"]].addItem(
                        f"{row['championName']} — {row['winrate_percent']:.1f}% ({int(row['games'])})"
                    )
            else:

                def to_tier_kda(k):
                    return (
                        "S"
                        if k >= 3.5
                        else ("A" if k >= 3.0 else ("B" if k >= 2.5 else "C"))
                    )

                data["Tier"] = data["kda_mean"].apply(to_tier_kda)
                data = data.sort_values(
                    ["Tier", "kda_mean", "games"], ascending=[True, False, False]
                )
                for _, row in data.iterrows():
                    self.role_lists[role][row["Tier"]].addItem(
                        f"{row['championName']} — KDA {row['kda_mean']:.2f} ({int(row['games'])})"
                    )

            counts = {t: self.role_lists[role][t].count() for t in ("S", "A", "B", "C")}
            total = sum(counts.values())
            self.role_info[role].setText(
                f"{role.capitalize()} — {total} champions  |  S:{counts['S']}  A:{counts['A']}  B:{counts['B']}  C:{counts['C']}"
            )

    def on_champion_clicked(self, role: str, item):
        if self.df_std is None:
            return
        champ = item.text().split("—", 1)[0].strip()
        df = self.df_std
        if "matchId" not in df.columns or "teamId" not in df.columns:
            QMessageBox.information(
                self, "Matchups", "matchId/teamId absents dans le CSV."
            )
            return
        sub = df[(df["role"] == role) & (df["championName"] == champ)].copy()
        if sub.empty:
            return
        df2 = df.copy()
        df2["kda_per_game"] = (df2["kills"] + df2["assists"]) / df2["deaths"].clip(
            lower=1
        )
        idx = df2.set_index(["matchId", "role"]).sort_index()
        rec = []
        for _, row in sub.iterrows():
            mid = row.get("matchId")
            my_team = row.get("teamId")
            if pd.isna(mid) or pd.isna(my_team):
                continue
            try:
                group = idx.loc[(mid, role)]
                if isinstance(group, pd.Series):
                    group = group.to_frame().T
            except KeyError:
                continue
            opp_rows = group[group["teamId"] != my_team]
            if opp_rows.empty:
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
        mu = pd.DataFrame(rec, columns=["opponent", "win", "K", "D", "A", "kda"])
        agg = mu.groupby("opponent", as_index=False).agg(
            games=("win", "size"),
            wins=("win", "sum"),
            K=("K", "sum"),
            D=("D", "sum"),
            A=("A", "sum"),
            kda=("kda", "mean"),
        )
        agg["losses"] = agg["games"] - agg["wins"]
        agg["wr_percent"] = agg["wins"] / agg["games"] * 100.0
        agg = agg[
            ["opponent", "games", "wins", "losses", "wr_percent", "kda", "K", "D", "A"]
        ].sort_values(["wr_percent", "games"], ascending=[False, False])

        self.tblMatchups.setRowCount(0)
        self.tblMatchups.setRowCount(len(agg))
        for r, (_, row) in enumerate(agg.iterrows()):
            kdastr = f"{int(row['K'])}/{int(row['D'])}/{int(row['A'])}"
            vals = [
                str(row["opponent"]),
                str(int(row["games"])),
                f"{int(row['wins'])}-{int(row['losses'])}",
                f"{row['wr_percent']:.1f}",
                f"{row['kda']:.2f}",
                kdastr,
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (1, 3, 4):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tblMatchups.setItem(r, c, it)
        self.tblMatchups.resizeColumnsToContents()
        self.tblMatchups.horizontalHeader().setStretchLastSection(True)


# =============================================================================
# Overlay de mini-camembert
# =============================================================================


class PieOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
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
        top.addWidget(self.btnClose, 0, Qt.AlignRight)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

    def show_pie(self, title, labels, sizes, pos=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        colors = [BLUE, BLUE_L, ORANGE, "#7FB3D5", "#F7C87A", "#B2D7EF", "#FFD08A"]
        ax.pie(
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
    def __init__(self):
        super().__init__()
        self.dfu: Optional[pd.DataFrame] = None
        self.overlay = PieOverlay(self)
        self._pie_conn = None

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
        self.tblRoles.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tblRoles.itemSelectionChanged.connect(self.on_role_selected)
        self.tblRoles.setAlternatingRowColors(True)

        self.tblChamps = QTableWidget(0, 8)
        self.tblChamps.setHorizontalHeaderLabels(
            ["Champion", "Games", "Wins", "WR %", "KDA", "K", "D", "A"]
        )
        self.tblChamps.setEditTriggers(QTableWidget.NoEditTriggers)
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

    # ---- collecte perso ------------------------------------------------------

    def _get_api_key(self) -> Optional[str]:
        return self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY")

    def on_build(self):
        riot_id = self.lineRiotID.text().strip()
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
        qtxt = self.comboQueue.currentText()
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
        if dfu.empty:
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

    def on_load_csv(self):
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
        if not needed.issubset(df.columns):
            QMessageBox.critical(
                self,
                "CSV",
                "Colonnes requises: role, teamWin, championName, kills, deaths, assists",
            )
            return
        df["role"] = df["role"].astype(str).str.lower()
        df["teamWin"] = (
            df["teamWin"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        )
        for c in ["kills", "deaths", "assists"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        self.dfu = df
        self.update_user_views()

    # ---- vues & camembert ----------------------------------------------------

    def role_pie(self, role: str):
        self.fig.clear()
        if self.dfu is None or self.dfu.empty:
            self.canvas.draw()
            return
        sub = self.dfu[self.dfu["role"] == role]
        if sub.empty:
            self.canvas.draw()
            return

        counts = {}
        for champ, n in sub["championName"].value_counts().items():
            t = champion_primary_tag(str(champ))
            counts[t] = counts.get(t, 0) + n
        total = sum(counts.values())
        if total == 0:
            self.canvas.draw()
            return
        labels = list(counts.keys())
        sizes = list(counts.values())
        ax = self.fig.add_subplot(111)
        ax.clear()
        colors = [BLUE, BLUE_L, ORANGE, "#7FB3D5", "#F7C87A", "#B2D7EF", "#FFD08A"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors[: len(labels)],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        )
        ax.set_title(f"Répartition des types — {role.capitalize()}", color=BLUE)
        for w in wedges:
            w.set_picker(True)
        old = getattr(self, "_pie_conn", None)
        if old is not None:
            self.canvas.mpl_disconnect(old)
        self._pie_conn = self.canvas.mpl_connect(
            "pick_event",
            lambda ev, role=role, labels=labels, sizes=sizes, sub=sub: self.on_pie_pick(
                ev, role, labels, sizes, sub
            ),
        )
        self.canvas.draw()

    def on_pie_pick(self, event, role, labels, sizes, sub_df):
        wedge = event.artist
        ax = event.canvas.figure.gca()
        wedges = [p for p in ax.patches if isinstance(p, type(wedge))]
        if wedge not in wedges:
            return
        idx = wedges.index(wedge)
        tag = labels[idx]
        vc = sub_df["championName"].value_counts()
        items = []
        for name, n in vc.items():
            if champion_primary_tag(str(name)) == tag:
                items.append((str(name), int(n)))
        if not items:
            return
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:8]
        rest = sum(n for _, n in items[8:])
        if rest > 0:
            top.append(("Autres", rest))
        clabels = [k for k, _ in top]
        csizes = [v for _, v in top]
        self.overlay.show_pie(f"{tag} — {role}", clabels, csizes, None)

    def on_role_selected(self):
        items = self.tblRoles.selectedItems()
        if not items:
            return
        row = items[0].row()
        role = self.tblRoles.item(row, 0).text()
        self.role_pie(role)

    def update_user_views(self):
        if self.dfu is None or self.dfu.empty:
            self.lblSummary.setText("—")
            self.tblRoles.setRowCount(0)
            self.tblChamps.setRowCount(0)
            self.fig.clear()
            self.canvas.draw()
            return
        ov = compute_player_overview(self.dfu)
        self.lblSummary.setText(
            f"Games: {ov['games']} | Wins: {ov['wins']} | WR: {ov['wr']:.1f}% | KDA: {ov['kda']:.2f}"
        )

        rdf = compute_player_by_role(self.dfu)
        self.tblRoles.setRowCount(0)
        self.tblRoles.setRowCount(len(rdf))
        for r, (_, row) in enumerate(rdf.iterrows()):
            vals = [
                row["role"],
                int(row["games"]),
                int(row["wins"]),
                f"{row['wr%']:.1f}",
                f"{row['kda']:.2f}",
                int(row["K"]),
                int(row["D"]),
                int(row["A"]),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c in (1, 3, 4, 5, 6, 7):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tblRoles.setItem(r, c, it)
        self.tblRoles.resizeColumnsToContents()
        self.tblRoles.horizontalHeader().setStretchLastSection(True)

        cdf = compute_player_by_champ(self.dfu)
        sk = self.comboSortChamp.currentText()
        if sk == "Winrate":
            cdf = cdf.sort_values(["wr%", "games"], ascending=[False, False])
        elif sk == "Games":
            cdf = cdf.sort_values("games", ascending=False)
        elif sk == "KDA":
            cdf = cdf.sort_values("kda", ascending=False)
        elif sk == "Kills":
            cdf = cdf.sort_values("K", ascending=False)
        elif sk == "Deaths":
            cdf = cdf.sort_values("D", ascending=True)
        elif sk == "Assists":
            cdf = cdf.sort_values("A", ascending=False)
        self.tblChamps.setRowCount(0)
        self.tblChamps.setRowCount(len(cdf))
        for r, (_, row) in enumerate(cdf.iterrows()):
            vals = [
                row["champion"],
                int(row["games"]),
                int(row["wins"]),
                f"{row['wr%']:.1f}",
                f"{row['kda']:.2f}",
                int(row["K"]),
                int(row["D"]),
                int(row["A"]),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c in (1, 3, 4, 5, 6, 7):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tblChamps.setItem(r, c, it)
        self.tblChamps.resizeColumnsToContents()
        self.tblChamps.horizontalHeader().setStretchLastSection(True)
        if len(rdf) > 0:
            self.tblRoles.selectRow(0)
            self.on_role_selected()


# =============================================================================
# Collecte perso depuis Riot (profil)
# =============================================================================


def normalize_riot_id(s: str) -> tuple[str, str] | None:
    if not s:
        return None
    s = re.sub(r"\s+", "", s.strip())
    if "#" not in s:
        return None
    g, t = s.split("#", 1)
    if not g or not t:
        return None
    return g, t.upper()


def riotid_to_puuid(rw: RiotWatcher, region: str, rid: str) -> Optional[str]:
    tup = normalize_riot_id(rid)
    if not tup:
        return None
    game, tag = tup
    acc = safe_call(rw.account.by_riot_id, region, game, tag)
    return acc.get("puuid")


def collect_user_participants(
    api_key: str,
    region: str,
    platform: str,
    riot_id: str,
    target_matches: int = 50,
    queue_id: Optional[int] = 420,
) -> pd.DataFrame:
    rw = RiotWatcher(api_key)
    lol = LolWatcher(api_key)
    puuid = riotid_to_puuid(rw, region, riot_id)
    if not puuid:
        raise ValueError("Riot ID introuvable")
    kw = {}
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
    rows = []
    for mid in mlist[:target_matches]:
        try:
            match = safe_call(lol.match.by_id, region, mid)
        except ApiError:
            continue
        info = match.get("info", {}) or {}
        parts = info.get("participants", []) or []
        me = None
        for p in parts:
            if p.get("puuid") == puuid:
                me = p
                break
        if not me:
            continue
        role = ROLE_MAP.get((me.get("teamPosition") or "").upper())
        if not role:
            continue
        my_team = me.get("teamId")
        opp_name = None
        for p in parts:
            if (
                ROLE_MAP.get((p.get("teamPosition") or "").upper()) == role
                and p.get("teamId") != my_team
            ):
                opp_name = p.get("championName")
                break
        k = int(me.get("kills", 0))
        d = int(me.get("deaths", 0))
        a = int(me.get("assists", 0))
        kda = (k + a) / (d if d > 0 else 1)
        rows.append(
            {
                "matchId": match.get("metadata", {}).get("matchId"),
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SuperLOL Assistant")
        self.setMinimumSize(1100, 720)
        self.is_dark_mode = False  # 添加模式标志
        self.ingame_tab = InGameTab()

        banner = QWidget()
        banner.setObjectName("banner")
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(8)
        logo_path = self._find_logo()
        if logo_path:
            pm = QPixmap(logo_path).scaledToHeight(54, Qt.SmoothTransformation)
            pic = QLabel()
            pic.setPixmap(pm)
            bl.addWidget(pic, 0, Qt.AlignLeft | Qt.AlignVCenter)
            self.setWindowIcon(QIcon(logo_path))
        titleWrap = QHBoxLayout()
        lab1 = QLabel("SuperLOL")
        lab1.setObjectName("brandTitle")
        lab2 = QLabel("Assistant")
        lab2.setObjectName("brandSubtitle")
        titleWrap.addWidget(lab1)
        titleWrap.addWidget(lab2)
        titleWrap.addStretch(1)

        # 添加昼夜模式切换按钮旁的热键配置按钮
        self.btnTheme = QPushButton("🌙 dark~")
        self.btnTheme.setFixedWidth(100)
        self.btnTheme.clicked.connect(self.toggle_theme)

        self.btnHotkey = QPushButton("设定发送键")
        self.btnHotkey.setFixedWidth(100)
        self.btnHotkey.clicked.connect(self.configure_hotkey)

        titleWrap.addWidget(self.btnTheme, 0, Qt.AlignRight | Qt.AlignVCenter)
        titleWrap.addWidget(self.btnHotkey, 0, Qt.AlignRight | Qt.AlignVCenter)

        bl.addLayout(titleWrap, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(MetaTab(), "META (global)")
        self.tabs.addTab(ProfileTab(), "MON PROFIL")
        self.tabs.addTab(self.ingame_tab, "IN GAME")
        # 默认热键为空
        self.hotkey_char: Optional[str] = None

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(banner)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def toggle_theme(self):
        """切换昼夜模式"""
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            QApplication.instance().setStyleSheet(APP_QSS_DARK)
            self.btnTheme.setText("☀️ dayday~")
        else:
            QApplication.instance().setStyleSheet(APP_QSS_LIGHT)
            self.btnTheme.setText("🌙 dark~")

    def _find_logo(self) -> Optional[str]:
        for name in ("superlol_logo.png", "logo.png", "superlol.png"):
            p = Path(name)
            if p.exists():
                return str(p)
        return None

    def configure_hotkey(self):
        """让玩家选择触发键，用于发送 InGameTab 的最新对局摘要。"""
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

        def on_trigger():
            txt = self.ingame_tab.get_live_summary_text()
            return txt or ""

        trashbase.start_hotkey_listener(self.hotkey_char, on_trigger)
        QMessageBox.information(
            self, "热键已设定", f"按下 '{self.hotkey_char}' 将发送当前局内扫描摘要。"
        )


# =============================================================================
# main
# =============================================================================


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS_LIGHT)  # 默认浅色模式
    w = MainWindow()
    w.resize(1280, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
