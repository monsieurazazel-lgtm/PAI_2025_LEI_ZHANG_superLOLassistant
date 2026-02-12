from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import pandas as pd
import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from riotwatcher import ApiError, LolWatcher, RiotWatcher
except Exception as e:
    raise SystemExit(
        "Installe d'abord : pip install riotwatcher pandas PySide6 matplotlib requests\n"
        + str(e)
    )

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------

PLATFORM_GROUPS: Dict[str, list[str]] = {
    "europe": ["euw1", "eun1", "tr1", "ru"],
    "americas": ["na1", "br1", "la1", "la2"],
    "asia": ["kr", "jp1"],
    "sea": ["oc1", "ph2", "sg2", "th2", "tw2", "vn2"],
}

ALL_PLATFORMS: list[str] = sorted(
    {p for group in PLATFORM_GROUPS.values() for p in group}
)
ALL_ROUTINGS: list[str] = ["europe", "americas", "asia", "sea"]

SLEEP_PER_CALL: float = 1.25
BACKOFF_429: float = 3.0

LIVECLIENT_URL: str = "http://127.0.0.1:2999/liveclientdata/allgamedata"

T = TypeVar("T")


def sleep_brief() -> None:
    """Sleep a short amount of time to reduce rate-limit issues."""
    time.sleep(SLEEP_PER_CALL)


def safe_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call a RiotWatcher function with a minimal retry strategy.

    Behavior:
        - If Riot returns HTTP 429 (rate-limit), waits BACKOFF_429 and retries.
        - If Riot returns HTTP 401/403, stops immediately (invalid/expired API key).
        - Otherwise, re-raises the original exception.

    Args:
        fn: Callable RiotWatcher endpoint (e.g. lol.match.by_id).
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        The value returned by fn(*args, **kwargs).

    Raises:
        SystemExit: If API key is invalid/expired (401/403).
        ApiError: For other Riot API errors.
    """
    while True:
        try:
            res = fn(*args, **kwargs)
            sleep_brief()
            return res
        except ApiError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)

            if status == 429:
                time.sleep(BACKOFF_429)
                continue

            if status in (401, 403):
                raise RuntimeError("Clé API invalide/expirée (401/403).")

            raise


def normalize_riot_id(text: str) -> Optional[Tuple[str, str]]:
    """Parse a Riot ID 'gameName#tagLine' into (gameName, tagLine).

    Notes:
        - Keeps internal spaces in gameName.
        - Accepts spaces around '#'.

    Args:
        text: Riot ID entered by the user.

    Returns:
        Tuple (gameName, tagLine) if valid, otherwise None.
    """
    if not text:
        return None

    text = text.strip()
    parts = re.split(r"\s*#\s*", text, maxsplit=1)
    if len(parts) != 2:
        return None

    game = parts[0].strip()
    tag = parts[1].strip().upper()
    if not game or not tag:
        return None

    return game, tag


def riotid_to_puuid(rw: RiotWatcher, routing: str, riot_id: str) -> Optional[str]:
    """Resolve Riot ID -> PUUID using Account API.

    Args:
        rw: RiotWatcher account client.
        routing: One of 'europe', 'americas', 'asia', 'sea'.
        riot_id: Riot ID formatted as 'gameName#tagLine'.

    Returns:
        PUUID if found, otherwise None.
    """
    parsed = normalize_riot_id(riot_id)
    if not parsed:
        return None

    game, tag = parsed
    acc = safe_call(rw.account.by_riot_id, routing, game, tag)
    acc_dict = cast(dict[str, Any], acc)
    return cast(Optional[str], acc_dict.get("puuid"))


def riotid_to_puuid_any_routing(
    rw: RiotWatcher,
    riot_id: str,
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Resolve Riot ID -> PUUID by trying multiple routings.

    Why:
        Users can select the wrong routing in the UI; we try all routings, with
        'preferred' first when valid.

    Args:
        rw: RiotWatcher account client.
        riot_id: Riot ID formatted as 'gameName#tagLine'.
        preferred: Routing to try first if valid.

    Returns:
        PUUID if found, otherwise None.
    """
    candidates: List[str] = []
    if preferred in ALL_ROUTINGS:
        candidates.append(preferred)
    candidates.extend([r for r in ALL_ROUTINGS if r not in candidates])

    for routing in candidates:
        try:
            pu = riotid_to_puuid(rw, routing, riot_id)
            if pu:
                return pu
        except Exception:
            continue

    return None


def liveclient_probe(timeout: float = 0.7) -> Tuple[Optional[dict[str, Any]], str]:
    """Probe the local LiveClient endpoint.

    Args:
        timeout: Network timeout in seconds.

    Returns:
        A tuple (payload, debug_text):
            - payload is a dict if LiveClient is reachable and contains players
            - payload is None otherwise
    """
    try:
        r = requests.get(LIVECLIENT_URL, timeout=timeout)
        if not r.ok:
            return None, f"LiveClient HTTP {r.status_code}"

        js = cast(dict[str, Any], r.json())
        all_players = js.get("allPlayers")
        if not all_players:
            return None, "LiveClient: allPlayers vide (pas en game ?)"

        return js, "LiveClient: OK"
    except Exception as e:
        return None, f"LiveClient: indisponible ({e})"


def weighted_note(
    champ_wr: float,
    global_wr: float,
    champ_games: int,
    total_games: int,
    total_requested: int,
) -> float:
    """Compute a weighted score from champion/global winrate with a confidence factor.

    Strategy:
        - Champion winrate slightly outweighs global winrate
        - Confidence increases with:
            (a) absolute champion games (saturating)
            (b) relative share among requested matches
        - Final note blends between champion-based score and global winrate using confidence

    Args:
        champ_wr: Champion winrate percentage.
        global_wr: Global winrate percentage.
        champ_games: Number of games on the champion within the sample.
        total_games: Total games within the sample.
        total_requested: Match count requested (spinbox value).

    Returns:
        A float score (same scale as winrate).
    """
    if total_games <= 0:
        return 0.0

    base = 0.65 * champ_wr + 0.35 * global_wr

    conf_abs = 1.0 - (2.71828 ** (-champ_games / 12.0))

    denom = max(1, int(total_requested))
    ratio = max(0.0, min(1.0, champ_games / denom))
    conf_rel = ratio**0.5

    conf = max(0.0, min(1.0, conf_abs * conf_rel))
    return float(conf * base + (1.0 - conf) * global_wr)


def account_routing_from_match_routing(routing: str) -> str:
    """Convert match routing to account routing.

    Note:
        Account-v1 does not accept 'sea', so we fallback to 'americas'.

    Args:
        routing: Match routing string.

    Returns:
        A routing compatible with account-v1.
    """
    return routing if routing in ("europe", "americas", "asia") else "americas"


def active_platform_for_lol(rw: RiotWatcher, routing: str, puuid: str) -> Optional[str]:
    """Get the official LoL platform (euw1/na1/kr/...) from active_shard.

    Args:
        rw: RiotWatcher account client.
        routing: Match routing from UI.
        puuid: Player PUUID.

    Returns:
        Platform in lowercase if found, otherwise None.
    """
    try:
        acc_routing = account_routing_from_match_routing(routing)
        shard = safe_call(rw.account.active_shard, acc_routing, "lol", puuid)
        shard_dict = cast(dict[str, Any], shard)

        plat = shard_dict.get("activeShard") or shard_dict.get("active_shard")
        return str(plat).lower() if plat else None
    except Exception:
        return None


def load_champion_id_to_name(
    lang: str = "en_US",
    cache_path: str = "data/champion_id_to_name.json",
) -> Dict[int, str]:
    """Build a mapping {championId(int): championName(str)} using Data Dragon.

    Caching:
        - If cache exists and is valid JSON, it is used.
        - Otherwise, the latest Data Dragon version is fetched and the cache is written.

    Args:
        lang: Data Dragon locale (e.g. 'en_US').
        cache_path: Local JSON cache path.

    Returns:
        A dict mapping champion numeric IDs to display names.
        Returns an empty dict if network is unavailable.
    """
    p = Path(cache_path)

    try:
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
    except Exception:
        pass

    version: Optional[str] = None
    try:
        r = requests.get(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=2.0
        )
        if r.ok:
            version = cast(list[Any], r.json() or [None])[0]
    except Exception:
        version = None

    if not version:
        return {}

    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{lang}/champion.json"
    out: Dict[int, str] = {}

    try:
        r = requests.get(url, timeout=2.5)
        if not r.ok:
            return {}

        js = cast(dict[str, Any], r.json() or {})
        champs = cast(dict[str, Any], js.get("data") or {}).values()
        for c in champs:
            try:
                cdict = cast(dict[str, Any], c)

                key_raw = cdict.get("key")
                if key_raw is None:
                    continue

                try:
                    cid = int(key_raw)
                except (TypeError, ValueError):
                    continue

                name_raw = cdict.get("name") or cdict.get("id") or str(cid)
                out[cid] = str(name_raw)
            except Exception:
                continue
    except Exception:
        return {}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {str(k): v for k, v in out.items()}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    return out


class InGameTab(QWidget):
    """UI tab that scans the current game and displays per-player stats.

    Flow:
        1) Try local LiveClient endpoint (best when you are currently in game)
        2) Otherwise fallback to Spectator API:
            - spectator-v5 via requests (PUUID)
            - if that fails, spectator-v4 via riotwatcher (summonerId)
    """

    COL_HEADERS = [
        "Summoner",
        "Champion",
        "Champ WR%",
        "Champ Games",
        "Global WR%",
        "KDA",
        "Games",
        "Note",
    ]
    COL_WIDTHS = [145, 110, 75, 80, 80, 60, 55, 60]

    def __init__(self) -> None:
        """Initialize widgets, caches, and layout."""
        super().__init__()

        self.cache_stats: Dict[str, dict[str, Any]] = {}
        self.cache_puuid_by_name: Dict[str, str] = {}

        self.last_rows_ally: List[list[str]] = []
        self.last_rows_enemy: List[list[str]] = []
        self.last_summary: str = ""

        self.lineApiKey = QLineEdit(os.getenv("RIOT_API_KEY") or "")
        self.lineRiotID = QLineEdit()
        self.lineRiotID.setPlaceholderText("Votre Riot ID (gameName#tagLine)")

        self.comboRegion = QComboBox()
        self.comboRegion.addItems(ALL_ROUTINGS)

        self.comboPlatform = QComboBox()
        self.comboPlatform.addItems(["auto"] + ALL_PLATFORMS)

        self.comboQueue = QComboBox()
        self.comboQueue.addItems(["0 (Toutes)", "420 (SoloQ)", "440 (Flex)"])

        self.spinRecent = QSpinBox()
        self.spinRecent.setRange(5, 100)
        self.spinRecent.setValue(20)

        self.btnScan = QPushButton("Scanner la partie en cours")
        self.btnScan.setObjectName("primaryBtn")
        self.btnScan.clicked.connect(self.scan_live)

        self.tblAllies = self._make_table()
        self.tblEnemies = self._make_table()

        self.champ_id_to_name = load_champion_id_to_name(lang="en_US")

        self._build_layout()

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------

    def _make_table(self) -> QTableWidget:
        """Create a read-only, compact QTableWidget configured for this tab."""
        tbl = QTableWidget(0, len(self.COL_HEADERS))
        tbl.setHorizontalHeaderLabels(self.COL_HEADERS)

        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setAlternatingRowColors(True)

        self._make_table_compact(tbl)
        return tbl

    def _build_layout(self) -> None:
        """Build the layout: form on top, two tables below (allies/enemies)."""
        form = QGridLayout()
        form.addWidget(QLabel("API Key"), 0, 0)
        form.addWidget(self.lineApiKey, 0, 1, 1, 3)
        form.addWidget(QLabel("Riot ID"), 1, 0)
        form.addWidget(self.lineRiotID, 1, 1, 1, 3)
        form.addWidget(QLabel("Region"), 2, 0)
        form.addWidget(self.comboRegion, 2, 1)
        form.addWidget(QLabel("Platform"), 2, 2)
        form.addWidget(self.comboPlatform, 2, 3)
        form.addWidget(QLabel("Queue (stats)"), 3, 0)
        form.addWidget(self.comboQueue, 3, 1)
        form.addWidget(QLabel("Recent games"), 3, 2)
        form.addWidget(self.spinRecent, 3, 3)
        form.addWidget(self.btnScan, 4, 0, 1, 4)

        top = QGroupBox("In Game — LiveClient (local) puis Spectator (Riot API)")
        top_layout = QVBoxLayout()
        top_layout.addLayout(form)
        top.setLayout(top_layout)

        g_allies = QGroupBox("Alliés")
        allies_layout = QVBoxLayout()
        allies_layout.addWidget(self.tblAllies)
        g_allies.setLayout(allies_layout)

        g_enemies = QGroupBox("Ennemis")
        enemies_layout = QVBoxLayout()
        enemies_layout.addWidget(self.tblEnemies)
        g_enemies.setLayout(enemies_layout)

        split = QHBoxLayout()
        split.addWidget(g_allies, 1)
        split.addWidget(g_enemies, 1)

        main = QVBoxLayout()
        main.addWidget(top)
        main.addLayout(split, 1)
        self.setLayout(main)

    def _make_table_compact(self, tbl: QTableWidget) -> None:
        """Apply a stable table configuration (prevents resize jitter/lag)."""
        tbl.setWordWrap(False)
        tbl.setTextElideMode(Qt.TextElideMode.ElideRight)
        tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        header = tbl.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setStretchLastSection(False)

        font = tbl.font()
        font.setPointSize(max(8, font.pointSize() - 1))
        tbl.setFont(font)

        tbl.verticalHeader().setDefaultSectionSize(22)

        for i, w in enumerate(self.COL_WIDTHS[: tbl.columnCount()]):
            tbl.setColumnWidth(i, w)

    def _fill_table(self, tbl: QTableWidget, rows: List[list[str]]) -> None:
        """Fill the table without autosize (keeps UI responsive)."""
        tbl.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                if c >= 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                tbl.setItem(r, c, item)

    def _set_scan_ui(self, running: bool) -> None:
        """Disable scan button during a scan to avoid concurrent requests."""
        try:
            self.btnScan.setEnabled(not running)
            self.btnScan.setText("Scan…" if running else "Scanner la partie en cours")
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Summary text (optional)
    # -----------------------------------------------------------------

    def _build_summary_text(
        self,
        allies: List[list[str]],
        enemies: List[list[str]],
        top_n: int = 3,
    ) -> str:
        """Build a short text summary for quick copy/paste.

        Args:
            allies: Sorted ally rows.
            enemies: Sorted enemy rows.
            top_n: Number of top entries to include per side.

        Returns:
            A single-line summary.
        """

        def fmt(side: List[list[str]]) -> str:
            parts: list[str] = []
            for row in side[:top_n]:
                parts.append(f"{row[0]}({row[1]}) {row[2]}%/{row[4]}% N{row[-1]}")
            return "; ".join(parts)

        if not allies and not enemies:
            return ""
        return f"Allies: {fmt(allies)} | Enemies: {fmt(enemies)}"

    def _build_funny_trash_text(
        self, allies: List[list[str]], enemies: List[list[str]]
    ) -> str:
        """Build a playful summary string (kept from original behavior)."""

        def note_val(row: list[str]) -> float:
            try:
                return float(row[-1])
            except Exception:
                return 0.0

        def side_stats(
            side: List[list[str]],
        ) -> tuple[Optional[list[str]], Optional[list[str]], float]:
            if not side:
                return None, None, 0.0
            return side[0], side[-1], sum(note_val(r) for r in side)

        if not allies and not enemies:
            return ""

        a_top, a_low, a_sum = side_stats(allies)
        e_top, _e_low, e_sum = side_stats(enemies)

        def nick(row: Optional[list[str]]) -> str:
            return f"{row[0]}({row[1]})" if row else "未知"

        parts = [
            f"己方总分{a_sum:.0f} vs 敌方{e_sum:.0f}",
            f"MVP预定：{nick(a_top)}" if a_top else "",
            f"需要照顾：{nick(a_low)}" if a_low else "",
            f"盯防对面：{nick(e_top)}" if e_top else "",
            "开局稳住，我们能赢！",
        ]
        return "，".join([p for p in parts if p])[:220]

    def get_live_summary_text(self) -> str:
        """Expose the last computed summary for external hotkeys."""
        if not self.last_rows_ally and not self.last_rows_enemy:
            return self._build_summary_text(self.last_rows_ally, self.last_rows_enemy)
        return self._build_funny_trash_text(self.last_rows_ally, self.last_rows_enemy)

    # -----------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------

    def champ_to_name(self, ch: Any) -> str:
        """Convert a champion identifier to a displayable name.

        Args:
            ch: Champion id (int/str digit) or champion name.

        Returns:
            A displayable champion name.
        """
        s = str(ch).strip()
        if not s:
            return ""
        if not s.isdigit():
            return s
        return self.champ_id_to_name.get(int(s), s)

    def _parse_queue(self, queue_txt: str) -> int:
        """Parse queue selection from UI into numeric queue id.

        Args:
            queue_txt: Combo text (e.g. '420 (SoloQ)').

        Returns:
            Queue id (0, 420, or 440).
        """
        if queue_txt.startswith("0"):
            return 0
        if queue_txt.startswith("420"):
            return 420
        return 440

    # -----------------------------------------------------------------
    # Riot API clients
    # -----------------------------------------------------------------

    def _api(self) -> Tuple[RiotWatcher, LolWatcher]:
        """Create Riot API clients from the current API key.

        Returns:
            (RiotWatcher, LolWatcher)

        Raises:
            RuntimeError: If API key is missing.
        """
        api_key = self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY")
        if not api_key:
            raise RuntimeError("API key manquante.")
        return RiotWatcher(api_key), LolWatcher(api_key)

    def _platform_candidates(self, platform_first: str) -> List[str]:
        """Return platform candidates to try (supports 'auto').

        Args:
            platform_first: Preferred platform from UI.

        Returns:
            A list of platforms to try, starting with platform_first when relevant.
        """
        if platform_first and platform_first != "auto":
            return [platform_first] + [p for p in ALL_PLATFORMS if p != platform_first]
        return list(ALL_PLATFORMS)

    def _summoner_by_puuid_multi(
        self,
        lol: LolWatcher,
        platform_first: str,
        puuid: str,
    ) -> Tuple[Optional[dict[str, Any]], Optional[str]]:
        """Try multiple platforms to resolve a summoner via PUUID."""
        for plat in self._platform_candidates(platform_first):
            try:
                data = safe_call(lol.summoner.by_puuid, plat, puuid)
                d = cast(dict[str, Any], data)
                if d.get("id"):
                    return d, plat
            except Exception:
                continue
        return None, None

    def _summoner_by_name_multi(
        self,
        lol: LolWatcher,
        platform_first: str,
        summoner_name: str,
    ) -> Tuple[Optional[dict[str, Any]], Optional[str]]:
        """Try multiple platforms to resolve a summoner via name.

        Note:
            Some riotwatcher versions have incomplete type hints for by_name.
            We use getattr to keep runtime compatibility and satisfy Pyright.
        """
        by_name = getattr(lol.summoner, "by_name", None)
        if by_name is None:
            return None, None

        for plat in self._platform_candidates(platform_first):
            try:
                data = safe_call(cast(Callable[..., Any], by_name), plat, summoner_name)
                d = cast(dict[str, Any], data)
                if d.get("id"):
                    return d, plat
            except Exception:
                continue
        return None, None

    def _active_game_spectator(
        self, lol: LolWatcher, platform: str, summ_id: str
    ) -> Optional[dict[str, Any]]:
        """Call spectator-v4 endpoint via riotwatcher with compatibility fallback."""
        fn = getattr(
            lol.spectator,
            "active_game_by_summoner",
            getattr(lol.spectator, "by_summoner", None),
        )
        if fn is None:
            return None
        try:
            data = safe_call(cast(Callable[..., Any], fn), platform, summ_id)
            return cast(dict[str, Any], data)
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def _stats_for(
        self,
        lol: LolWatcher,
        routing: str,
        puuid: str,
        champ_name: str,
        count: int,
        queue: int,
    ) -> dict[str, Any]:
        """Compute global stats and champion-specific stats from recent matches.

        Implementation details:
            - Fetch match ids with matchlist_by_puuid
            - For each match, fetch match details with by_id
            - Extract the participant block matching the given puuid
            - Aggregate wins and K/D/A
            - Compute both global and champion-filtered metrics

        Args:
            lol: LolWatcher client.
            routing: Match routing (europe/americas/asia/sea).
            puuid: Target player PUUID.
            champ_name: Champion name to filter stats.
            count: Number of matches requested (clamped to 100).
            queue: Queue id (0 = all, 420/440).

        Returns:
            A dict with keys:
                - games (int)
                - wr (float)
                - kda (float)
                - champ_games (int)
                - champ_wr (float)
                - champ_kda (float)
        """
        cache_key = f"{routing}|{puuid}|{queue}|{count}|{champ_name}"
        if cache_key in self.cache_stats:
            return self.cache_stats[cache_key]

        params: dict[str, Any] = {"count": min(100, count)}
        if queue in (420, 440):
            params["queue"] = queue  # already implies ranked queues

        match_ids = (
            safe_call(lol.match.matchlist_by_puuid, routing, puuid, **params) or []
        )
        match_ids_list = cast(list[Any], match_ids)

        records: List[dict[str, Any]] = []

        for mid in match_ids_list:
            try:
                match = safe_call(lol.match.by_id, routing, mid)
                match_dict = cast(dict[str, Any], match)
            except Exception:
                continue

            info = cast(dict[str, Any], match_dict.get("info", {}) or {})
            participants = cast(list[Any], info.get("participants", []) or [])
            for p in participants:
                pdict = cast(dict[str, Any], p)
                if pdict.get("puuid") == puuid:
                    records.append(
                        {
                            "win": bool(pdict.get("win")),
                            "champ": pdict.get("championName"),
                            "k": int(pdict.get("kills", 0)),
                            "d": int(pdict.get("deaths", 0)),
                            "a": int(pdict.get("assists", 0)),
                        }
                    )
                    break

        df = pd.DataFrame(records)

        out: dict[str, Any] = {
            "games": 0,
            "wr": 0.0,
            "kda": 0.0,
            "champ_games": 0,
            "champ_wr": 0.0,
            "champ_kda": 0.0,
        }

        if not df.empty:
            games = int(len(df))
            wr = float(df["win"].mean() * 100.0)
            kda = float((df["k"].sum() + df["a"].sum()) / max(1, df["d"].sum()))

            cdf = df[df["champ"] == champ_name]
            champ_games = int(len(cdf))
            champ_wr = float((cdf["win"].mean() * 100.0) if champ_games > 0 else 0.0)
            champ_kda = float(
                ((cdf["k"].sum() + cdf["a"].sum()) / max(1, cdf["d"].sum()))
                if champ_games > 0
                else 0.0
            )

            out = {
                "games": games,
                "wr": wr,
                "kda": kda,
                "champ_games": champ_games,
                "champ_wr": champ_wr,
                "champ_kda": champ_kda,
            }

        self.cache_stats[cache_key] = out
        return out

    # -----------------------------------------------------------------
    # Row formatting
    # -----------------------------------------------------------------

    def _row_from_stats(
        self,
        lol: LolWatcher,
        routing: str,
        puuid: Optional[str],
        name: str,
        champ: str,
        recent: int,
        queue: int,
    ) -> list[str]:
        """Build a UI row (8 columns) from player identity + computed stats.

        Args:
            lol: LolWatcher client.
            routing: Match routing.
            puuid: Player PUUID (None allowed, returns default zeros).
            name: Display name.
            champ: Champion name.
            recent: Number of recent games to scan.
            queue: Queue filter.

        Returns:
            A list[str] representing the table row.
        """
        if not puuid:
            return [name, champ, "0.0", "0", "0.0", "0.00", "0", "0.0"]

        stat = self._stats_for(lol, routing, puuid, champ, recent, queue)
        champ_wr = float(stat.get("champ_wr", 0.0))
        global_wr = float(stat.get("wr", 0.0))
        champ_games = int(stat.get("champ_games", 0))
        games = int(stat.get("games", 0))
        kda = float(stat.get("kda", 0.0))

        note = weighted_note(champ_wr, global_wr, champ_games, games, recent)
        return [
            name,
            champ,
            f"{champ_wr:.1f}",
            str(champ_games),
            f"{global_wr:.1f}",
            f"{kda:.2f}",
            str(games),
            f"{note:.1f}",
        ]

    def _commit_rows(
        self, rows_ally: List[list[str]], rows_enemy: List[list[str]]
    ) -> None:
        """Sort rows by note and update UI + cached last results."""
        rows_ally.sort(key=lambda x: float(x[-1]), reverse=True)
        rows_enemy.sort(key=lambda x: float(x[-1]), reverse=True)

        self._fill_table(self.tblAllies, rows_ally)
        self._fill_table(self.tblEnemies, rows_enemy)

        self.last_rows_ally = rows_ally
        self.last_rows_enemy = rows_enemy
        self.last_summary = self._build_funny_trash_text(rows_ally, rows_enemy)

    # -----------------------------------------------------------------
    # LiveClient helpers
    # -----------------------------------------------------------------

    def _guess_platform_from_me(
        self,
        rw: RiotWatcher,
        lol: LolWatcher,
        routing_ui: str,
        platform_ui: str,
        my_riotid: str,
    ) -> str:
        """Determine a LoL platform when UI is set to 'auto'.

        Strategy:
            - If UI is not auto, return it.
            - Otherwise, try RiotID -> PUUID -> summoner.by_puuid across platforms.
            - Final fallback: 'euw1'.

        Args:
            rw: RiotWatcher account client.
            lol: LolWatcher client.
            routing_ui: Routing from UI.
            platform_ui: Platform from UI ('auto' allowed).
            my_riotid: Riot ID entered by user.

        Returns:
            A platform string (e.g. 'euw1').
        """
        if platform_ui not in ("", "auto"):
            return platform_ui

        plat_guess: Optional[str] = None
        if my_riotid:
            pu = riotid_to_puuid_any_routing(rw, my_riotid, preferred=routing_ui)
            if pu:
                summ, plat = self._summoner_by_puuid_multi(lol, "auto", pu)
                if summ and plat:
                    plat_guess = plat

        platform = plat_guess or "euw1"
        idx = self.comboPlatform.findText(platform)
        if idx >= 0:
            self.comboPlatform.setCurrentIndex(idx)
        return platform

    def _puuid_from_liveclient_player(
        self,
        rw: RiotWatcher,
        lol: LolWatcher,
        routing_ui: str,
        platform: str,
        pobj: dict[str, Any],
    ) -> Optional[str]:
        """Resolve a LiveClient player object into a PUUID.

        Order:
            1) If LiveClient exposes riotId 'game#tag', resolve via Account API.
            2) Otherwise, fallback to summonerName via summoner.by_name.

        Cache:
            Uses (platform|summonerName) cache to avoid repeated requests.

        Args:
            rw: RiotWatcher account client.
            lol: LolWatcher client.
            routing_ui: Routing from UI.
            platform: Selected platform.
            pobj: LiveClient player dict.

        Returns:
            PUUID if resolved, otherwise None.
        """
        rid = pobj.get("riotId") or pobj.get("riotID") or pobj.get("riotid")
        if isinstance(rid, str) and "#" in rid:
            try:
                pu = riotid_to_puuid_any_routing(rw, rid, preferred=routing_ui)
                if pu:
                    return pu
            except Exception:
                pass

        name = str(pobj.get("summonerName") or "").strip()
        if not name:
            return None

        cache_key = f"{platform}|{name.lower()}"
        if cache_key in self.cache_puuid_by_name:
            return self.cache_puuid_by_name[cache_key]

        by_name = getattr(lol.summoner, "by_name", None)
        if by_name is None:
            return None

        try:
            summ = safe_call(cast(Callable[..., Any], by_name), platform, name)
            summ_dict = cast(dict[str, Any], summ)
            pu = cast(Optional[str], summ_dict.get("puuid"))
            if pu:
                self.cache_puuid_by_name[cache_key] = pu
                return pu
        except Exception:
            return None

        return None

    # -----------------------------------------------------------------
    # Spectator helpers
    # -----------------------------------------------------------------

    def _spectator_v5_active_game(
        self,
        platform: str,
        puuid: str,
        api_key: str,
    ) -> Tuple[Optional[dict[str, Any]], str]:
        """Call spectator-v5 (PUUID) directly via HTTP requests.

        Args:
            platform: LoL platform (e.g. 'euw1').
            puuid: Player PUUID.
            api_key: Riot API key.

        Returns:
            A tuple (game_json, debug_text).
        """
        url = (
            f"https://{platform}.api.riotgames.com/"
            f"lol/spectator/v5/active-games/by-summoner/{puuid}"
        )
        try:
            r = requests.get(url, headers={"X-Riot-Token": api_key}, timeout=1.8)
            if r.status_code == 404:
                return None, "SpectatorV5: 404 (pas en game / non observable)"
            if not r.ok:
                return None, f"SpectatorV5: HTTP {r.status_code} -> {r.text[:180]}"
            return cast(dict[str, Any], r.json()), "SpectatorV5: OK"
        except Exception as e:
            return None, f"SpectatorV5: erreur réseau ({e})"

    def _platform_from_recent_match(self, lol: LolWatcher, puuid: str) -> Optional[str]:
        """Guess platform from the prefix of a recent match id.

        Args:
            lol: LolWatcher client.
            puuid: Player PUUID.

        Returns:
            Platform prefix (e.g. 'euw1') if found, otherwise None.
        """
        for rt in ALL_ROUTINGS:
            try:
                ids = safe_call(lol.match.matchlist_by_puuid, rt, puuid, count=1) or []
                ids_list = cast(list[Any], ids)
                if not ids_list:
                    continue
                mid = str(ids_list[0])
                if "_" in mid:
                    return mid.split("_", 1)[0].lower()
            except Exception:
                continue
        return None

    def _puuid_from_summoner_id(
        self, lol: LolWatcher, platform: str, summoner_id: str
    ) -> Optional[str]:
        """Fallback resolution: summonerId -> PUUID via summoner.by_id."""
        try:
            s = safe_call(lol.summoner.by_id, platform, summoner_id)
            sdict = cast(dict[str, Any], s)
            return cast(Optional[str], sdict.get("puuid"))
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Main scan
    # -----------------------------------------------------------------

    def scan_live(self) -> None:
        """Run a full scan and update UI (LiveClient first, then Spectator)."""
        try:
            self._set_scan_ui(running=True)

            rw, lol = self._api()
            api_key = (
                self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY") or ""
            ).strip()
            if not api_key:
                raise RuntimeError("API key manquante.")

            routing_ui = self.comboRegion.currentText().strip().lower()
            platform_ui = self.comboPlatform.currentText().strip().lower()
            my_riotid = self.lineRiotID.text().strip()

            queue = self._parse_queue(self.comboQueue.currentText())
            recent = int(self.spinRecent.value())

            lc, lc_dbg = liveclient_probe()
            if lc is not None:
                self._scan_from_liveclient(
                    rw,
                    lol,
                    routing_ui,
                    platform_ui,
                    my_riotid,
                    queue,
                    recent,
                    lc,
                    lc_dbg,
                )
                return

            self._scan_from_spectator(
                rw, lol, api_key, routing_ui, platform_ui, my_riotid, queue, recent
            )

        except Exception as e:
            QMessageBox.critical(self, "In Game", f"Erreur: {e}")
        finally:
            self._set_scan_ui(running=False)

    def _scan_from_liveclient(
        self,
        rw: RiotWatcher,
        lol: LolWatcher,
        routing_ui: str,
        platform_ui: str,
        my_riotid: str,
        queue: int,
        recent: int,
        lc: dict[str, Any],
        lc_dbg: str,
    ) -> None:
        """Scan from LiveClient data and update tables."""
        players = cast(list[Any], lc.get("allPlayers") or [])
        if not players:
            QMessageBox.information(
                self, "In Game", "LiveClient OK mais allPlayers vide.\n" + lc_dbg
            )
            return

        platform = self._guess_platform_from_me(
            rw, lol, routing_ui, platform_ui, my_riotid
        )

        rows_ally: List[list[str]] = []
        rows_enemy: List[list[str]] = []

        for pobj in players:
            p = cast(dict[str, Any], pobj)
            name = str(p.get("summonerName") or "")
            champ = self.champ_to_name(
                p.get("championName") or p.get("championId") or ""
            )
            team = str(p.get("team") or "")  # ORDER / CHAOS

            puuid = self._puuid_from_liveclient_player(rw, lol, routing_ui, platform, p)
            row = self._row_from_stats(
                lol, routing_ui, puuid, name, champ, recent, queue
            )

            (rows_ally if team == "ORDER" else rows_enemy).append(row)

        self._commit_rows(rows_ally, rows_enemy)
        QMessageBox.information(
            self, "In Game", f"✅ Partie trouvée via LiveClient.\n{lc_dbg}"
        )

    def _scan_from_spectator(
        self,
        rw: RiotWatcher,
        lol: LolWatcher,
        api_key: str,
        routing_ui: str,
        platform_ui: str,
        my_riotid: str,
        queue: int,
        recent: int,
    ) -> None:
        """Scan from Spectator API and update tables."""
        if not my_riotid:
            QMessageBox.warning(
                self, "In Game", "Saisis un Riot ID (gameName#tagLine)."
            )
            return

        puuid = riotid_to_puuid_any_routing(rw, my_riotid, preferred=routing_ui)
        if not puuid:
            QMessageBox.warning(self, "In Game", "Riot ID introuvable (Account API).")
            return

        platform_found = platform_ui if platform_ui not in ("", "auto") else ""
        if not platform_found:
            platform_found = active_platform_for_lol(rw, routing_ui, puuid) or ""

        if not platform_found:
            platform_found = self._platform_from_recent_match(lol, puuid) or ""

        if not platform_found:
            _summ, plat = self._summoner_by_puuid_multi(lol, "auto", puuid)
            platform_found = plat or ""

        if not platform_found:
            QMessageBox.warning(
                self,
                "In Game",
                "Riot ID valide, mais impossible de déterminer la plateforme LoL (euw1/na1/kr…).",
            )
            return

        idx = self.comboPlatform.findText(platform_found)
        if idx >= 0:
            self.comboPlatform.setCurrentIndex(idx)

        game, dbg = self._spectator_v5_active_game(platform_found, puuid, api_key)

        if not game:
            summ, _ = self._summoner_by_puuid_multi(lol, platform_found, puuid)
            summ_id = cast(Optional[str], (summ or {}).get("id"))
            if summ_id:
                game2 = self._active_game_spectator(lol, platform_found, summ_id)
                if game2:
                    game = game2
                    dbg = "SpectatorV4: OK (fallback)"

        if not game:
            QMessageBox.information(
                self, "In Game", f"Pas de partie en cours (Spectator).\n{dbg}"
            )
            return

        rows_ally: List[list[str]] = []
        rows_enemy: List[list[str]] = []

        participants = cast(list[Any], game.get("participants", []) or [])
        for pobj in participants:
            p = cast(dict[str, Any], pobj)
            name = p.get("summonerName") or p.get("riotId") or ""
            team_id = p.get("teamId")
            champ = self.champ_to_name(
                p.get("championName") or p.get("championId") or ""
            )

            ppu = cast(Optional[str], p.get("puuid"))
            if not ppu and p.get("summonerId"):
                ppu = self._puuid_from_summoner_id(
                    lol, platform_found, str(p.get("summonerId"))
                )

            row = self._row_from_stats(
                lol, routing_ui, ppu, str(name), str(champ), recent, queue
            )
            (rows_ally if team_id == 100 else rows_enemy).append(row)

        self._commit_rows(rows_ally, rows_enemy)
        QMessageBox.information(
            self,
            "In Game",
            f"✅ Partie trouvée via Spectator ({platform_found}).\n{dbg}",
        )
