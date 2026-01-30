from __future__ import annotations

import os
import time
import re
from typing import Dict, Optional, Tuple, List
import json
from pathlib import Path

import pandas as pd
import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QHeaderView,
)

try:
    from riotwatcher import LolWatcher, RiotWatcher, ApiError
except Exception as e:
    raise SystemExit(
        "Installe d'abord : pip install riotwatcher pandas PySide6 matplotlib requests\n" + str(e)
    )

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
PLATFORM_GROUPS: Dict[str, list[str]] = {
    "europe":   ["euw1", "eun1", "tr1", "ru"],
    "americas": ["na1", "br1", "la1", "la2"],
    "asia":     ["kr", "jp1"],
    "sea":      ["oc1", "ph2", "sg2", "th2", "tw2", "vn2"],
}

ALL_PLATFORMS = sorted({p for g in PLATFORM_GROUPS.values() for p in g})
ALL_ROUTINGS = ["europe", "americas", "asia", "sea"]

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


def normalize_riot_id(s: str) -> Optional[Tuple[str, str]]:
    """
    Retourne (gameName, TAG) ou None.
    IMPORTANT: ne PAS supprimer les espaces à l'intérieur du gameName.
    On tolère des espaces autour du '#'.
    """
    if not s:
        return None

    s = s.strip()
    # split tolérant: "Expedition 28 #0911" -> ("Expedition 28", "0911")
    m = re.split(r"\s*#\s*", s, maxsplit=1)
    if len(m) != 2:
        return None

    game = m[0].strip()
    tag = m[1].strip().upper()
    if not game or not tag:
        return None
    return game, tag


def riotid_to_puuid(rw: RiotWatcher, routing: str, rid: str) -> Optional[str]:
    tup = normalize_riot_id(rid)
    if not tup:
        return None
    game, tag = tup
    acc = safe_call(rw.account.by_riot_id, routing, game, tag)
    return acc.get("puuid")


def riotid_to_puuid_any_routing(
    rw: RiotWatcher, rid: str, preferred: Optional[str] = None
) -> Optional[str]:
    """
    Teste toutes les routes (europe/americas/asia/sea) pour résoudre RiotID -> PUUID.
    On teste d'abord 'preferred' si fourni.
    """
    routings = []
    if preferred in ALL_ROUTINGS:
        routings.append(preferred)
    routings += [r for r in ALL_ROUTINGS if r not in routings]

    for r in routings:
        try:
            pu = riotid_to_puuid(rw, r, rid)
            if pu:
                return pu
        except Exception:
            continue
    return None


def liveclient_probe(timeout: float = 0.7) -> Tuple[Optional[dict], str]:
    """
    Essaye le endpoint local LoL (quand LoL est ouvert et en game).
    Retourne (json, debug_text) ou (None, debug_text)
    """
    url = "http://127.0.0.1:2999/liveclientdata/allgamedata"
    try:
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            return None, f"LiveClient HTTP {r.status_code}"
        js = r.json()
        if not js.get("allPlayers"):
            return None, "LiveClient: allPlayers vide (pas en game ?)"
        return js, "LiveClient: OK"
    except Exception as e:
        return None, f"LiveClient: indisponible ({e})"


def weighted_note(champ_wr: float, global_wr: float, champ_games: int, total_games: int, total_requested: int) -> float:
    """
    Note pondérée :
    - WR champion a un peu plus de poids que WR global
    - la confiance dépend à la fois :
        (1) du nb de games sur le champion (champ_games)
        (2) du ratio champ_games / total_requested (ex: 2/5 > 2/20)
    """
    if total_games <= 0:
        return 0.0

    w_champ = 0.65
    w_global = 0.35
    base = w_champ * champ_wr + w_global * global_wr

    # confiance "absolue" (augmente avec champ_games)
    conf_abs = 1.0 - (2.71828 ** (-champ_games / 12.0))

    # confiance "relative" (2/5 > 2/20), on prend sqrt pour pas tuer trop fort la note
    denom = max(1, int(total_requested))
    ratio = max(0.0, min(1.0, champ_games / denom))
    conf_rel = ratio ** 0.5

    conf = max(0.0, min(1.0, conf_abs * conf_rel))

    out = conf * base + (1.0 - conf) * global_wr
    return float(out)


def account_routing_from_match_routing(routing: str) -> str:
    # Account-v1 n'a que: americas / asia / europe (pas "sea") :contentReference[oaicite:3]{index=3}
    return routing if routing in ("europe", "americas", "asia") else "americas"


def active_platform_for_lol(rw: RiotWatcher, routing: str, puuid: str) -> Optional[str]:
    """
    Donne la platform LoL (euw1/na1/kr/...) via account-v1 active shard.
    riotwatcher expose rw.account.active_shard(region, game, puuid). :contentReference[oaicite:4]{index=4}
    """
    try:
        acc_routing = account_routing_from_match_routing(routing)
        shard = safe_call(rw.account.active_shard, acc_routing, "lol", puuid)
        # selon lib/retour, la clé est souvent "activeShard"
        plat = (shard or {}).get("activeShard") or (shard or {}).get("active_shard")
        return str(plat).lower() if plat else None
    except Exception:
        return None

def load_champion_id_to_name(lang: str = "en_US", cache_path: str = "data/champion_id_to_name.json") -> Dict[int, str]:
    """
    Construit un dict {championId(int): championName(str)} depuis Data Dragon champion.json.
    Cache sur disque pour éviter de retélécharger.
    """
    p = Path(cache_path)
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in data.items()}
    except Exception:
        pass

    # Data Dragon: récupérer la dernière version + champion.json :contentReference[oaicite:1]{index=1}
    ver = None
    try:
        r = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=2.0)
        if r.ok:
            ver = (r.json() or [None])[0]
    except Exception:
        ver = None

    if not ver:
        return {}

    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/champion.json"
    out: Dict[int, str] = {}
    try:
        r = requests.get(url, timeout=2.5)
        if not r.ok:
            return {}
        js = r.json() or {}
        champs = (js.get("data") or {}).values()
        for c in champs:
            # champion.json contient "key" (id numérique en string) + "name" :contentReference[oaicite:2]{index=2}
            try:
                cid = int(c.get("key"))
                out[cid] = c.get("name") or c.get("id") or str(cid)
            except Exception:
                continue
    except Exception:
        return {}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out


class InGameTab(QWidget):

    def __init__(self):
        

        super().__init__()
        self.cache_stats: Dict[str, dict] = {}
        self.cache_puuid_by_name: Dict[str, str] = {}

        self.last_rows_ally = []
        self.last_rows_enemy = []
        self.last_summary = ""

        self.lineApiKey = QLineEdit(os.getenv("RIOT_API_KEY") or "")
        self.lineRiotID = QLineEdit()
        self.lineRiotID.setPlaceholderText("Votre Riot ID (gameName#tagLine)")

        self.comboRegion = QComboBox()
        self.comboRegion.addItems(["europe", "americas", "asia", "sea"])

        self.comboPlatform = QComboBox()
        # ✅ IMPORTANT : on ajoute "auto" (sinon ton code ne peut pas être robuste)
        self.comboPlatform.addItems(["auto"] + ALL_PLATFORMS)

        self.comboQueue = QComboBox()
        self.comboQueue.addItems(["0 (Toutes)", "420 (SoloQ)", "440 (Flex)"])

        self.spinRecent = QSpinBox()
        self.spinRecent.setRange(5, 100)
        self.spinRecent.setValue(20)

        self.btnScan = QPushButton("Scanner la partie en cours")
        self.btnScan.setObjectName("primaryBtn")
        self.btnScan.clicked.connect(self.scan_live)

        # 7 colonnes (comme ton UI)
        self.tblAllies = QTableWidget(0, 8)
        self.tblAllies.setHorizontalHeaderLabels(
            ["Summoner", "Champion", "Champ WR%", "Champ Games", "Global WR%", "KDA", "Games", "Note"]
        )
        self.tblAllies.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tblAllies.setAlternatingRowColors(True)
        self.tblAllies.horizontalHeader().setStretchLastSection(True)

        self.tblEnemies = QTableWidget(0, 8)
        self.tblEnemies.setHorizontalHeaderLabels(
            ["Summoner", "Champion", "Champ WR%", "Champ Games", "Global WR%", "KDA", "Games", "Note"]
        )
        self.tblEnemies.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tblEnemies.setAlternatingRowColors(True)
        self.tblEnemies.horizontalHeader().setStretchLastSection(True)
       
        self._make_table_compact(self.tblAllies)
        self._make_table_compact(self.tblEnemies)

        self.champ_id_to_name = load_champion_id_to_name(lang="en_US")

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
        v = QVBoxLayout()
        v.addLayout(form)
        top.setLayout(v)

        split = QHBoxLayout()
        g1 = QGroupBox("Alliés")
        a = QVBoxLayout()
        a.addWidget(self.tblAllies)
        g1.setLayout(a)

        g2 = QGroupBox("Ennemis")
        b = QVBoxLayout()
        b.addWidget(self.tblEnemies)
        g2.setLayout(b)

        split.addWidget(g1, 1)
        split.addWidget(g2, 1)

        main = QVBoxLayout()
        main.addWidget(top)
        main.addLayout(split, 1)
        self.setLayout(main)

    def _make_table_compact(self, tbl: QTableWidget):
        tbl.setWordWrap(False)
        tbl.setTextElideMode(Qt.ElideRight)          # coupe avec "..."
        tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # header fixed + tailles manuelles
        h = tbl.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Fixed)
        h.setStretchLastSection(False)

        # un peu plus petit visuellement
        f = tbl.font()
        f.setPointSize(max(8, f.pointSize() - 1))
        tbl.setFont(f)

        vh = tbl.verticalHeader()
        vh.setDefaultSectionSize(22)

        # Largeurs (8 colonnes) : ajuste si tu veux
        widths = [145, 110, 75, 80, 80, 60, 55, 60]  # Summoner, Champion, ChampWR, ChampGames, GlobalWR, KDA, Games, Note
        for i, w in enumerate(widths[:tbl.columnCount()]):
            tbl.setColumnWidth(i, w)

    # ------------------------------------------------------------
    # Texte récupérable par interface.py (hotkey)
    # ------------------------------------------------------------
    def _build_summary_text(self, allies, enemies, top_n: int = 3) -> str:
        def fmt(side):
            parts = []
            for row in side[:top_n]:
                # row: [name, champ, champ_wr, champ_games, global_wr, kda, games, note]
                parts.append(f"{row[0]}({row[1]}) {row[2]}%/{row[4]}% N{row[-1]}")
            return "; ".join(parts)

        if not allies and not enemies:
            return ""
        return f"Allies: {fmt(allies)} | Enemies: {fmt(enemies)}"

        # helpers champ id -> name (si tu as déjà un mapping, garde-le; sinon affichera championName)
    def champ_to_name(self, ch):
        s = str(ch).strip()
        if not s:
            return ""
        if not s.isdigit():          # déjà un nom
            return s
        return self.champ_id_to_name.get(int(s), s)


    def _build_funny_trash_text(self, allies, enemies) -> str:
        def safe_note(x):
            try:
                return float(x[-1])
            except Exception:
                return 0.0

        def side_stats(side):
            if not side:
                return None, None, 0.0
            top = side[0]
            low = side[-1]
            total = sum(safe_note(r) for r in side)
            return top, low, total

        if not allies and not enemies:
            return ""

        a_top, a_low, a_sum = side_stats(allies)
        e_top, e_low, e_sum = side_stats(enemies)

        def nick(row):
            return f"{row[0]}({row[1]})" if row else "未知"

        msg_parts = []
        msg_parts.append(f"己方总分{a_sum:.0f} vs 敌方{e_sum:.0f}")
        if a_top:
            msg_parts.append(f"MVP预定：{nick(a_top)}")
        if a_low:
            msg_parts.append(f"需要照顾：{nick(a_low)}")
        if e_top:
            msg_parts.append(f"盯防对面：{nick(e_top)}")
        msg_parts.append("开局稳住，我们能赢！")
        return "，".join(msg_parts)[:220]

    def get_live_summary_text(self) -> str:
        if not self.last_rows_ally and not self.last_rows_enemy:
            return self._build_summary_text(self.last_rows_ally, self.last_rows_enemy)
        return self._build_funny_trash_text(self.last_rows_ally, self.last_rows_enemy)

    # ------------------------------------------------------------
    # Riot API
    # ------------------------------------------------------------
    def _api(self):
        api_key = self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY")
        if not api_key:
            raise RuntimeError("API key manquante.")
        return RiotWatcher(api_key), LolWatcher(api_key)

    def _summoner_by_puuid_multi(
        self, lol: LolWatcher, platform_first: str, puuid: str
    ) -> Tuple[Optional[dict], Optional[str]]:
        candidates: List[str] = []
        if platform_first and platform_first != "auto":
            candidates.append(platform_first)
        candidates += [p for p in ALL_PLATFORMS if p not in candidates]

        for plat in candidates:
            try:
                data = safe_call(lol.summoner.by_puuid, plat, puuid)
                if data and data.get("id"):
                    return data, plat
            except Exception:
                continue
        return None, None

    def _summoner_by_name_multi(
        self, lol: LolWatcher, platform_first: str, summoner_name: str
    ) -> Tuple[Optional[dict], Optional[str]]:
        candidates: List[str] = []
        if platform_first and platform_first != "auto":
            candidates.append(platform_first)
        candidates += [p for p in ALL_PLATFORMS if p not in candidates]

        for plat in candidates:
            try:
                data = safe_call(lol.summoner.by_name, plat, summoner_name)
                if data and data.get("id"):
                    return data, plat
            except Exception:
                continue
        return None, None

    def _resolve_player(self, rw: RiotWatcher, lol: LolWatcher, riotid: str, preferred_routing: str, platform_ui: str):
        puuid = riotid_to_puuid_any_routing(rw, riotid, preferred=preferred_routing)
        if not puuid:
            return None, None, None

        # ✅ NOUVEAU : récupérer la platform officielle
        plat = active_platform_for_lol(rw, preferred_routing, puuid)

        # si on a une platform fiable, on l'utilise directement
        if plat:
            try:
                summ = safe_call(lol.summoner.by_puuid, plat, puuid)
                if summ and summ.get("id"):
                    return puuid, summ.get("id"), plat
            except Exception:
                pass

        # sinon fallback sur ton ancienne logique multi-plateformes
        summ, plat2 = self._summoner_by_puuid_multi(lol, platform_ui, puuid)
        if summ and summ.get("id"):
            return puuid, summ.get("id"), plat2

        # fallback by_name si besoin…
        tup = normalize_riot_id(riotid)
        game_name = tup[0] if tup else ""
        if game_name:
            summ2, plat3 = self._summoner_by_name_multi(lol, platform_ui, game_name)
            if summ2 and summ2.get("id"):
                return puuid, summ2.get("id"), plat3

        return puuid, None, None


    def _active_game_spectator(self, lol: LolWatcher, platform: str, summ_id: str) -> Optional[dict]:
        fn = getattr(lol.spectator, "active_game_by_summoner", getattr(lol.spectator, "by_summoner", None))
        if fn is None:
            return None
        try:
            return safe_call(fn, platform, summ_id)
        except Exception:
            return None

    def _stats_for(
        self,
        lol: LolWatcher,
        routing: str,
        puuid: str,
        champ_name: str,
        count: int,
        queue: int,
    ) -> dict:
        """
        Retourne:
          games, wr, kda, champ_games, champ_wr, champ_kda
        """
        key = f"{routing}|{puuid}|{queue}|{count}|{champ_name}"
        if key in self.cache_stats:
            return self.cache_stats[key]

        kw = {"count": min(100, count)}
        if queue in (420, 440):
            kw.update(queue=queue, type="ranked")

        mlist = safe_call(lol.match.matchlist_by_puuid, routing, puuid, **kw) or []
        rec = []
        for mid in mlist:
            try:
                match = safe_call(lol.match.by_id, routing, mid)
            except Exception:
                continue
            info = match.get("info", {}) or {}
            for p in info.get("participants", []) or []:
                if p.get("puuid") == puuid:
                    rec.append(
                        {
                            "win": bool(p.get("win")),
                            "champ": p.get("championName"),
                            "k": int(p.get("kills", 0)),
                            "d": int(p.get("deaths", 0)),
                            "a": int(p.get("assists", 0)),
                        }
                    )
                    break

        df = pd.DataFrame(rec)
        if df.empty:
            out = {
                "games": 0,
                "wr": 0.0,
                "kda": 0.0,
                "champ_games": 0,
                "champ_wr": 0.0,
                "champ_kda": 0.0,
            }
        else:
            games = len(df)
            wr = df["win"].mean() * 100.0
            kda = (df["k"].sum() + df["a"].sum()) / max(1, df["d"].sum())

            cdf = df[df["champ"] == champ_name]
            champ_games = int(len(cdf))
            champ_wr = (cdf["win"].mean() * 100.0) if champ_games > 0 else 0.0
            champ_kda = (
                (cdf["k"].sum() + cdf["a"].sum()) / max(1, cdf["d"].sum())
                if champ_games > 0
                else 0.0
            )

            out = {
                "games": games,
                "wr": float(wr),
                "kda": float(kda),
                "champ_games": champ_games,
                "champ_wr": float(champ_wr),
                "champ_kda": float(champ_kda),
            }

        self.cache_stats[key] = out
        return out

    # ------------------------------------------------------------
    # Scan principal
    # ------------------------------------------------------------
    def scan_live(self):
        """
        LiveClient (local) d'abord.
        Si KO -> Spectator (Riot API) via spectator-v5 (PUUID) + fallback spectator-v4.
        """
        try:
            try:
                self.btnScan.setEnabled(False)
                self.btnScan.setText("Scan…")
            except Exception:
                pass

            rw, lol = self._api()
            api_key = (self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError("API key manquante.")

            routing_ui = self.comboRegion.currentText().strip().lower()
            platform_ui = self.comboPlatform.currentText().strip().lower()
            my_riotid = self.lineRiotID.text().strip()

            queue_txt = self.comboQueue.currentText()
            queue = 0 if queue_txt.startswith("0") else (420 if queue_txt.startswith("420") else 440)
            recent = int(self.spinRecent.value())  # <-- total de matchs demandés

            def fill(tbl, rows):
                tbl.setRowCount(0)
                tbl.setRowCount(len(rows))
                for r, vals in enumerate(rows):
                    for c, v in enumerate(vals):
                        it = QTableWidgetItem(str(v))
                        if c >= 2:  # colonnes numériques
                            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        tbl.setItem(r, c, it)
                # surtout PAS resizeColumnsToContents()

            # ------------------------------------------------------------
            # 1) LiveClient FIRST
            # ------------------------------------------------------------
            lc, lc_dbg = liveclient_probe()
            if lc is not None:
                players = lc.get("allPlayers") or []
                if not players:
                    QMessageBox.information(self, "In Game", "LiveClient OK mais allPlayers vide.\n" + lc_dbg)
                    return

                # Déterminer platform si "auto" (pour résoudre by_name -> puuid)
                platform = platform_ui
                if platform in ("", "auto"):
                    plat_guess = None
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

                rows_ally, rows_enemy = [], []

                def puuid_from_liveclient_player(pobj: dict) -> Optional[str]:
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

                    ck = f"{platform}|{name.lower()}"
                    if ck in self.cache_puuid_by_name:
                        return self.cache_puuid_by_name[ck]

                    try:
                        s = safe_call(lol.summoner.by_name, platform, name)
                        pu = (s or {}).get("puuid")
                        if pu:
                            self.cache_puuid_by_name[ck] = pu
                            return pu
                    except Exception:
                        return None
                    return None



                for p in players:
                    name = str(p.get("summonerName") or "")
                    champ = self.champ_to_name(p.get("championName") or p.get("championId") or "")

                    team = str(p.get("team") or "")  # ORDER/CHAOS

                    puuid = puuid_from_liveclient_player(p)
                    if puuid:
                        stat = self._stats_for(lol, routing_ui, puuid, champ, recent, queue)
                        champ_wr = float(stat.get("champ_wr", 0.0))
                        global_wr = float(stat.get("wr", 0.0))
                        champ_games = int(stat.get("champ_games", 0))
                        games = int(stat.get("games", 0))
                        kda = float(stat.get("kda", 0.0))

                        note = weighted_note(champ_wr, global_wr, champ_games, games, recent)

                        row = [
                            name,
                            champ,
                            f"{champ_wr:.1f}",
                            str(champ_games),          # <-- Champ Games
                            f"{global_wr:.1f}",
                            f"{kda:.2f}",
                            str(games),
                            f"{note:.1f}",
                        ]
                    else:
                        row = [name, champ, "0.0", "0", "0.0", "0.00", "0", "0.0"]

                    (rows_ally if team == "ORDER" else rows_enemy).append(row)

                rows_ally.sort(key=lambda x: float(x[-1]), reverse=True)
                rows_enemy.sort(key=lambda x: float(x[-1]), reverse=True)
                fill(self.tblAllies, rows_ally)
                fill(self.tblEnemies, rows_enemy)

                self.last_rows_ally = rows_ally
                self.last_rows_enemy = rows_enemy
                self.last_summary = self._build_funny_trash_text(rows_ally, rows_enemy)

                QMessageBox.information(self, "In Game", f"✅ Partie trouvée via LiveClient.\n{lc_dbg}")
                return

            # ------------------------------------------------------------
            # 2) Spectator fallback
            # ------------------------------------------------------------
            if not my_riotid:
                QMessageBox.warning(self, "In Game", "Saisis un Riot ID (gameName#tagLine).")
                return

            puuid = riotid_to_puuid_any_routing(rw, my_riotid, preferred=routing_ui)
            if not puuid:
                QMessageBox.warning(self, "In Game", "Riot ID introuvable (Account API).")
                return

            platform_found = platform_ui if platform_ui not in ("", "auto") else ""
            if not platform_found:
                platform_found = active_platform_for_lol(rw, routing_ui, puuid) or ""

            if not platform_found:
                def find_platform_from_recent_match(puuid_: str) -> Optional[str]:
                    routings_to_try = ["europe", "americas", "asia", "sea"]
                    for rt in routings_to_try:
                        try:
                            ids = safe_call(lol.match.matchlist_by_puuid, rt, puuid_, count=1) or []
                            if not ids:
                                continue
                            mid = str(ids[0])
                            if "_" in mid:
                                return mid.split("_", 1)[0].lower()
                        except Exception:
                            continue
                    return None
                platform_found = find_platform_from_recent_match(puuid) or ""

            if not platform_found:
                _summ, plat = self._summoner_by_puuid_multi(lol, "auto", puuid)
                platform_found = plat or ""

            if not platform_found:
                QMessageBox.warning(self, "In Game",
                    "Riot ID valide, mais impossible de déterminer la plateforme LoL (euw1/na1/kr…).")
                return

            idx = self.comboPlatform.findText(platform_found)
            if idx >= 0:
                self.comboPlatform.setCurrentIndex(idx)

            def spectator_v5_active_game(platform: str, puuid_: str) -> Tuple[Optional[dict], str]:
                url = f"https://{platform}.api.riotgames.com/lol/spectator/v5/active-games/by-summoner/{puuid_}"
                try:
                    r = requests.get(url, headers={"X-Riot-Token": api_key}, timeout=1.8)
                    if r.status_code == 404:
                        return None, "SpectatorV5: 404 (pas en game / non observable)"
                    if not r.ok:
                        return None, f"SpectatorV5: HTTP {r.status_code} -> {r.text[:180]}"
                    return r.json(), "SpectatorV5: OK"
                except Exception as e:
                    return None, f"SpectatorV5: erreur réseau ({e})"

            game, dbg = spectator_v5_active_game(platform_found, puuid)

            if not game:
                summ, _ = self._summoner_by_puuid_multi(lol, platform_found, puuid)
                summ_id = (summ or {}).get("id") if summ else None
                if summ_id:
                    game = self._active_game_spectator(lol, platform_found, summ_id)
                    if game:
                        dbg = "SpectatorV4: OK (fallback)"

            if not game:
                QMessageBox.information(self, "In Game", f"Pas de partie en cours (Spectator).\n{dbg}")
                return

            rows_ally, rows_enemy = [], []
            parts = game.get("participants", []) or []

  

            for p in parts:
                name = p.get("summonerName") or p.get("riotId") or ""
                team_id = p.get("teamId")
                champ = self.champ_to_name(p.get("championName") or p.get("championId") or "")


                ppu = p.get("puuid")
                if not ppu and p.get("summonerId"):
                    try:
                        s = safe_call(lol.summoner.by_id, platform_found, p.get("summonerId"))
                        ppu = (s or {}).get("puuid")
                    except Exception:
                        ppu = None

                if ppu:
                    stat = self._stats_for(lol, routing_ui, ppu, str(champ), recent, queue)
                    champ_wr = float(stat.get("champ_wr", 0.0))
                    global_wr = float(stat.get("wr", 0.0))
                    champ_games = int(stat.get("champ_games", 0))
                    games = int(stat.get("games", 0))
                    kda = float(stat.get("kda", 0.0))

                    note = weighted_note(champ_wr, global_wr, champ_games, games, recent)

                    row = [
                        str(name),
                        str(champ),
                        f"{champ_wr:.1f}",
                        str(champ_games),
                        f"{global_wr:.1f}",
                        f"{kda:.2f}",
                        str(games),
                        f"{note:.1f}",
                    ]
                else:
                    row = [str(name), str(champ), "0.0", "0", "0.0", "0.00", "0", "0.0"]

                (rows_ally if team_id == 100 else rows_enemy).append(row)

            rows_ally.sort(key=lambda x: float(x[-1]), reverse=True)
            rows_enemy.sort(key=lambda x: float(x[-1]), reverse=True)
            fill(self.tblAllies, rows_ally)
            fill(self.tblEnemies, rows_enemy)

            self.last_rows_ally = rows_ally
            self.last_rows_enemy = rows_enemy
            self.last_summary = self._build_funny_trash_text(rows_ally, rows_enemy)

            QMessageBox.information(self, "In Game", f"✅ Partie trouvée via Spectator ({platform_found}).\n{dbg}")

        except Exception as e:
            QMessageBox.critical(self, "In Game", f"Erreur: {e}")
        finally:
            try:
                self.btnScan.setEnabled(True)
                self.btnScan.setText("Scanner la partie en cours")
            except Exception:
                pass
