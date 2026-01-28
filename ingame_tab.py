from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

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
)

try:
    from riotwatcher import LolWatcher, RiotWatcher, ApiError
except Exception as e:
    raise SystemExit(
        "Installe d'abord : pip install riotwatcher pandas PySide6 matplotlib requests\n"
        + str(e)
    )

# Local copy of minimal helpers to avoid circular imports
PLATFORM_GROUPS: Dict[str, list[str]] = {
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


def riotid_to_puuid(rw: RiotWatcher, region: str, rid: str) -> Optional[str]:
    import re

    def normalize_riot_id(s: str):
        if not s:
            return None
        s2 = re.sub(r"\s+", "", s.strip())
        if "#" not in s2:
            return None
        g, t = s2.split("#", 1)
        if not g or not t:
            return None
        return g, t.upper()

    tup = normalize_riot_id(rid)
    if not tup:
        return None
    game, tag = tup
    acc = safe_call(rw.account.by_riot_id, region, game, tag)
    return acc.get("puuid")


class InGameTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cache_stats: Dict[str, dict] = {}
        # 新增：缓存最近扫描结果
        self.last_rows_ally = []
        self.last_rows_enemy = []
        self.last_summary = ""
        self.lineApiKey = QLineEdit(os.getenv("RIOT_API_KEY") or "")
        self.lineRiotID = QLineEdit()
        self.lineRiotID.setPlaceholderText("Votre Riot ID (gameName#tagLine)")
        self.comboRegion = QComboBox()
        self.comboRegion.addItems(["europe", "americas", "asia", "sea"])
        self.comboPlatform = QComboBox()
        self.comboPlatform.addItems(
            [
                "euw1",
                "eune1",
                "tr1",
                "ru",
                "na1",
                "la1",
                "la2",
                "br1",
                "kr",
                "jp1",
                "oc1",
            ]
        )
        self.comboQueue = QComboBox()
        self.comboQueue.addItems(["0 (Toutes)", "420 (SoloQ)", "440 (Flex)"])
        self.spinRecent = QSpinBox()
        self.spinRecent.setRange(5, 100)
        self.spinRecent.setValue(20)
        self.btnScan = QPushButton("Scanner la partie en cours")
        self.btnScan.setObjectName("primaryBtn")
        self.btnScan.clicked.connect(self.scan_live)

        self.tblAllies = QTableWidget(0, 7)
        self.tblAllies.setHorizontalHeaderLabels(
            ["Summoner", "Champion", "Champ WR%", "Global WR%", "KDA", "Games", "Note"]
        )
        self.tblAllies.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tblAllies.setAlternatingRowColors(True)
        self.tblAllies.horizontalHeader().setStretchLastSection(True)
        self.tblEnemies = QTableWidget(0, 7)
        self.tblEnemies.setHorizontalHeaderLabels(
            ["Summoner", "Champion", "Champ WR%", "Global WR%", "KDA", "Games", "Note"]
        )
        self.tblEnemies.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tblEnemies.setAlternatingRowColors(True)
        self.tblEnemies.horizontalHeader().setStretchLastSection(True)

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
        top = QGroupBox(
            "In Game — scout automatique (PvP via Spectator, Quickplay/Bots via client local)"
        )
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

    def _build_summary_text(self, allies, enemies, top_n: int = 3) -> str:
        def fmt(side):
            parts = []
            for row in side[:top_n]:
                # row: [name, champ, champ_wr, wr, kda, games, note]
                parts.append(f"{row[0]}({row[1]}) {row[2]}%/{row[3]}% N{row[6]}")
            return "; ".join(parts)

        if not allies and not enemies:
            return ""
        return f"Allies: {fmt(allies)} | Enemies: {fmt(enemies)}"

    def get_live_summary_text(self) -> str:
        # 外部调用获取当前可发送文本
        if self.last_summary:
            return self.last_summary
        return self._build_summary_text(self.last_rows_ally, self.last_rows_enemy)

    def _api(self):
        api_key = self.lineApiKey.text().strip() or os.getenv("RIOT_API_KEY")
        if not api_key:
            raise RuntimeError("API key manquante.")
        return RiotWatcher(api_key), LolWatcher(api_key)

    def _summoner_by_puuid_multi(
        self, lol: LolWatcher, region: str, platform_first: str, puuid: str
    ) -> Optional[dict]:
        platforms = [platform_first] + [
            p for p in PLATFORM_GROUPS.get(region, []) if p != platform_first
        ]
        tried = set()
        for plat in platforms:
            if plat in tried:
                continue
            tried.add(plat)
            try:
                data = safe_call(lol.summoner.by_puuid, plat, puuid)
                if data and data.get("id"):
                    return data
            except ApiError:
                continue
            except Exception:
                continue
        return None

    def _summoner_id_from_riotid(
        self, rw: RiotWatcher, lol: LolWatcher, region: str, platform: str, riotid: str
    ) -> Tuple[Optional[str], Optional[str]]:
        puuid = riotid_to_puuid(rw, region, riotid)
        if not puuid:
            return (None, None)
        summ = self._summoner_by_puuid_multi(lol, region, platform, puuid)
        if not summ:
            return (None, puuid)
        return (summ.get("id"), puuid)

    def _active_game_spectator_multi(
        self, lol: LolWatcher, platform_first: str, region: str, summ_id: str
    ) -> Optional[dict]:
        fn = getattr(
            lol.spectator,
            "active_game_by_summoner",
            getattr(lol.spectator, "by_summoner", None),
        )
        if fn is None:
            return None
        tried = set()
        platforms = [platform_first] + [
            p for p in PLATFORM_GROUPS.get(region, []) if p != platform_first
        ]
        for plat in platforms:
            if plat in tried:
                continue
            tried.add(plat)
            try:
                game = safe_call(fn, plat, summ_id)
                if game:
                    return game
            except ApiError:
                continue
            except Exception:
                continue
        return None

    def _stats_for(
        self,
        lol: LolWatcher,
        region: str,
        puuid: str,
        champ_name: str,
        count: int,
        queue: int,
    ) -> dict:
        if puuid in self.cache_stats:
            return self.cache_stats[puuid]
        kw = {"count": min(100, count)}
        if queue in (420, 440):
            kw.update(queue=queue, type="ranked")
        mlist = safe_call(lol.match.matchlist_by_puuid, region, puuid, **kw) or []
        rec = []
        for mid in mlist:
            try:
                match = safe_call(lol.match.by_id, region, mid)
            except ApiError:
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
            out = {"games": 0, "wr": 0.0, "kda": 0.0, "champ_wr": 0.0}
        else:
            wr = df["win"].mean() * 100.0
            kda = (df["k"].sum() + df["a"].sum()) / max(1, df["d"].sum())
            cdf = df[df["champ"] == champ_name]
            champ_wr = (cdf["win"].mean() * 100.0) if len(cdf) > 0 else 0.0
            out = {"games": len(df), "wr": wr, "kda": kda, "champ_wr": champ_wr}
        self.cache_stats[puuid] = out
        return out

    def scan_live(self):
        try:
            rows_ally.sort(key=lambda x: float(x[-1]), reverse=True)
            rows_enemy.sort(key=lambda x: float(x[-1]), reverse=True)
            fill(self.tblAllies, rows_ally)
            fill(self.tblEnemies, rows_enemy)
            rw, lol = self._api()
            region = self.comboRegion.currentText().strip().lower()
            platform = self.comboPlatform.currentText().strip().lower()
            riotid = self.lineRiotID.text().strip()
            queue_txt = self.comboQueue.currentText()
            queue = (
                0
                if queue_txt.startswith("0")
                else (420 if queue_txt.startswith("420") else 440)
            )
            recent = int(self.spinRecent.value())

            summ_id, puuid = self._summoner_id_from_riotid(
                rw, lol, region, platform, riotid
            )
            if not puuid:
                QMessageBox.warning(self, "In Game", "ID introuvable (Account API).")
                return
            if not summ_id:
                QMessageBox.warning(
                    self,
                    "In Game",
                    "Riot ID valide, mais introuvable via by_puuid sur les plateformes testées. "
                    "Vérifie la plateforme (EUW1/EUNE1/NA1…) ou réessaie (DNS).",
                )
                return

            game = self._active_game_spectator_multi(lol, platform, region, summ_id)
            if game is None:
                try:
                    r = requests.get(
                        "http://127.0.0.1:2999/liveclientdata/allgamedata", timeout=0.7
                    )
                    if r.ok and r.json().get("allPlayers"):
                        game = {"liveclient": r.json()}
                except Exception:
                    game = None

            if not game:
                QMessageBox.information(self, "In Game", "Pas de partie en cours.")
                return

            rows_ally = []
            rows_enemy = []
            if "liveclient" in game:
                lc = game["liveclient"]
                players = lc.get("allPlayers") or []
                for p in players:
                    name = p.get("summonerName")
                    champ = p.get("championName")
                    team = p.get("team")
                    stat = {"games": 0, "wr": 0.0, "kda": 0.0, "champ_wr": 0.0}
                    note = 0.0
                    row = [
                        name,
                        champ,
                        f"{stat['champ_wr']:.1f}",
                        f"{stat['wr']:.1f}",
                        f"{stat['kda']:.2f}",
                        str(stat["games"]),
                        f"{note:.1f}",
                    ]
                    (rows_ally if team == "ORDER" else rows_enemy).append(row)
            else:
                parts = game.get("participants", []) or []
                for p in parts:
                    name = p.get("summonerName")
                    champ = p.get("championId") or p.get("championName")
                    team = p.get("teamId")
                    try:
                        summ = safe_call(
                            lol.summoner.by_id, platform, p.get("summonerId")
                        )
                        ppu = summ.get("puuid")
                    except ApiError:
                        ppu = None
                    if not ppu:
                        stat = {"games": 0, "wr": 0.0, "kda": 0.0, "champ_wr": 0.0}
                    else:
                        stat = self._stats_for(
                            lol, region, ppu, str(champ), recent, queue
                        )
                    note = (
                        (stat["champ_wr"] * 0.6 + stat["wr"] * 0.4)
                        if stat["games"] > 0
                        else 0.0
                    )
                    row = [
                        name,
                        str(champ),
                        f"{stat['champ_wr']:.1f}",
                        f"{stat['wr']:.1f}",
                        f"{stat['kda']:.2f}",
                        str(stat["games"]),
                        f"{note:.1f}",
                    ]
                    (rows_ally if team == 100 else rows_enemy).append(row)

            def fill(tbl, rows):
                tbl.setRowCount(0)
                tbl.setRowCount(len(rows))
                for r, vals in enumerate(rows):
                    for c, v in enumerate(vals):
                        it = QTableWidgetItem(v)
                        if c in (2, 3, 4, 5, 6):
                            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        tbl.setItem(r, c, it)
                tbl.resizeColumnsToContents()
                tbl.horizontalHeader().setStretchLastSection(True)

            rows_ally.sort(key=lambda x: float(x[-1]), reverse=True)
            rows_enemy.sort(key=lambda x: float(x[-1]), reverse=True)
            fill(self.tblAllies, rows_ally)
            fill(self.tblEnemies, rows_enemy)

            # 缓存结果用于热键发送
            self.last_rows_ally = rows_ally
            self.last_rows_enemy = rows_enemy
            self.last_summary = self._build_summary_text(rows_ally, rows_enemy)
        except Exception as e:
            QMessageBox.critical(self, "In Game", f"Erreur: {e}")
