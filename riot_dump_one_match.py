#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
riot_dump_one_match.py
Script pour extraire et formater les données d'un match League of Legends via l'API Riot.
"""

from __future__ import annotations
import argparse
import os
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Tentative d'importation de la bibliothèque riotwatcher
try:
    from riotwatcher import RiotWatcher, LolWatcher, ApiError
except Exception as e:
    raise SystemExit(
        "riotwatcher n'est pas installé. Fais: pip install riotwatcher\n" + str(e)
    )

# Dossier par défaut pour stocker les fichiers extraits
DATA_DIR = Path("data_one_match")
DATA_DIR.mkdir(exist_ok=True)


# ------------------------------
# Utils (Outils)
# ------------------------------

def save_json(path: Path, obj: Any) -> None:
    """Sauvegarde un objet Python dans un fichier au format JSON.

    Args:
        path: Chemin complet du fichier de destination.
        obj: L'objet (dictionnaire ou liste) à sauvegarder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def df_to_csv(path: Path, df: pd.DataFrame) -> None:
    """Enregistre un DataFrame Pandas au format CSV sans index.

    Args:
        path: Chemin du fichier CSV de sortie.
        df: Le DataFrame à convertir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def flatten_participants(info: Dict[str, Any]) -> pd.DataFrame:
    """Transforme les données imbriquées des participants en un tableau plat.

    Args:
        info: Dictionnaire 'info' extrait du JSON du match.

    Returns:
        Un DataFrame contenant les statistiques détaillées par joueur.
    """
    # Colonnes de base à extraire
    cols_basic = [
        "participantId", "teamId", "win", "summonerName", "puuid",
        "championName", "champLevel", "teamPosition", "role", "lane",
        "kills", "deaths", "assists", "totalMinionsKilled",
        "neutralMinionsKilled", "goldEarned", "goldSpent", "visionScore",
        "timeCCingOthers", "totalDamageDealt", "totalDamageDealtToChampions",
        "magicDamageDealt", "physicalDamageDealt", "trueDamageDealt",
        "damageDealtToObjectives", "damageSelfMitigated", "totalHeal",
        "totalHealsOnTeammates", "totalDamageTaken", "totalTimeSpentDead",
        "wardsPlaced", "wardsKilled",
    ]
    parts = info.get("participants", [])
    rows = []
    for p in parts:
        row = {k: p.get(k) for k in cols_basic}
        # Extraction des 7 emplacements d'objets (items)
        for i in range(7):
            row[f"item{i}"] = p.get(f"item{i}")
        # Sorts d'invocateur
        row["summoner1Id"] = p.get("summoner1Id")
        row["summoner2Id"] = p.get("summoner2Id")
        
        # Gestion des runes (perks)
        perks = p.get("perks", {})
        row["perks_raw"] = json.dumps(perks, ensure_ascii=False)
        try:
            styles = perks.get("styles", [])
            if styles and len(styles) >= 2:
                prim, sec = styles[0], styles[1]
                row["runes_primary_style"] = prim.get("style")
                row["runes_secondary_style"] = sec.get("style")
                sels = []
                for st in styles:
                    for sel in st.get("selections") or []:
                        sels.append(sel.get("perk"))
                for j, perk in enumerate(sels[:6]):
                    row[f"rune_{j + 1}"] = perk
        except Exception:
            pass
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Réorganisation des colonnes pour la lisibilité
    item_cols = [f"item{i}" for i in range(7)]
    rune_cols = [c for c in df.columns if c.startswith("rune_")] + [
        "runes_primary_style", "runes_secondary_style", "perks_raw"
    ]
    spell_cols = ["summoner1Id", "summoner2Id"]
    ordered = [c for c in cols_basic if c in df.columns] + item_cols + spell_cols + rune_cols
    return df.reindex(columns=ordered)


def flatten_teams(info: Dict[str, Any]) -> pd.DataFrame:
    """Extrait les statistiques globales des équipes.

    Args:
        info: Dictionnaire 'info' du match.

    Returns:
        DataFrame avec les victoires, bans et objectifs par équipe.
    """
    records = []
    for t in info.get("teams", []):
        rec = {
            "teamId": t.get("teamId"),
            "win": t.get("win"),
            "ban_0": None, "ban_1": None, "ban_2": None, "ban_3": None, "ban_4": None,
        }
        # Enregistrement des 5 champions bannis
        for i, b in enumerate(t.get("bans", [])[:5]):
            rec[f"ban_{i}"] = b.get("championId")
        # Extraction des objectifs (Dragon, Baron, etc.)
        obj = t.get("objectives", {}) or {}
        for key in ["baron", "dragon", "tower", "inhibitor", "riftHerald", "champion"]:
            o = obj.get(key, {})
            rec[f"{key}_first"] = o.get("first")
            rec[f"{key}_kills"] = o.get("kills")
        records.append(rec)
    return pd.DataFrame(records)


def flatten_objectives_long(info: Dict[str, Any]) -> pd.DataFrame:
    """Version 'longue' des objectifs pour faciliter l'analyse par type.

    Args:
        info: Dictionnaire 'info' du match.

    Returns:
        DataFrame où chaque ligne est un objectif spécifique d'une équipe.
    """
    records = []
    for t in info.get("teams", []):
        tid = t.get("teamId")
        for key, o in (t.get("objectives") or {}).items():
            records.append({
                "teamId": tid,
                "objective": key,
                "first": o.get("first"),
                "kills": o.get("kills"),
            })
    return pd.DataFrame(records)


def flatten_timeline_events(timeline: Dict[str, Any]) -> pd.DataFrame:
    """Extrait tous les événements chronologiques de la partie.

    Args:
        timeline: Dictionnaire complet de la timeline du match.

    Returns:
        DataFrame listant chaque événement avec son horodatage.
    """
    if not timeline:
        return pd.DataFrame()
    frames = timeline.get("info", {}).get("frames", [])
    recs = []
    for fr in frames:
        ts = fr.get("timestamp")
        for ev in fr.get("events", []):
            row = {"timestamp": ts, "type": ev.get("type")}
            # Champs d'intérêt à extraire dynamiquement
            fields = [
                "participantId", "killerId", "victimId", "assistingParticipantIds",
                "itemId", "skillSlot", "level", "levelUpType", "wardType",
                "creatorId", "teamId", "laneType", "buildingType", "towerType",
                "bounty", "multiKillLength", "monsterType", "monsterSubType",
                "position", "realTimestamp", "shutdownBounty", "killStreakLength"
            ]
            for k in fields:
                if k in ev: row[k] = ev.get(k)
            recs.append(row)
    
    df = pd.DataFrame(recs)
    # Conversion des dictionnaires/listes en chaînes JSON pour le stockage CSV
    for c in df.columns:
        if df[c].map(lambda x: isinstance(x, (dict, list))).any():
            df[c] = df[c].map(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
    return df


# ------------------------------
# Fetchers (Récupérateurs)
# ------------------------------

def get_puuid(
    rw: RiotWatcher, lol: LolWatcher, region: str, platform: str, game_name: str, tag_line: str
) -> str:
    """Récupère l'identifiant unique (PUUID) d'un joueur via son Riot ID.

    Args:
        rw: Instance pour l'API Account.
        lol: Instance pour l'API League.
        region: Région de routage (ex: 'europe').
        platform: Plateforme (ex: 'euw1').
        game_name: Nom du joueur.
        tag_line: Tag du joueur.

    Returns:
        Le PUUID du joueur.
    """
    try:
        acct = rw.account.by_riot_id(region, game_name, tag_line)
        puuid = acct.get("puuid")
        if puuid:
            print(f"[OK] PUUID via account-v1: {puuid}")
            return str(puuid)
    except ApiError as e:
        print(f"[WARN] account-v1 a échoué: {e}")

    # Fallback sur l'ancienne méthode summoner-v4
    summ = lol.summoner.by_name(platform, game_name)
    puuid = summ.get("puuid")
    print(f"[OK] PUUID via summoner-v4 fallback: {puuid}")
    return str(puuid)


def pick_match_id(
    lol: LolWatcher, region: str, puuid: str, queue: int = 420, count: int = 50
) -> str:
    """Récupère la liste des matchs récents et en choisit un au hasard.

    Args:
        lol: Instance LolWatcher.
        region: Région de routage.
        puuid: PUUID du joueur.
        queue: ID de la file (420 pour SoloQ).
        count: Nombre de matchs à scanner.

    Returns:
        Un identifiant de match (matchId).
    """
    mlist = lol.match.matchlist_by_puuid(
        region, puuid, type="ranked", queue=queue, count=count
    )
    if not mlist:
        raise SystemExit("Aucun match trouvé pour ce joueur.")
    mid = random.choice(mlist)
    print(f"[OK] Match choisi: {mid} (parmi {len(mlist)} IDs)")
    return str(mid)


# ------------------------------
# Main logic (Logique principale)
# ------------------------------

def main() -> None:
    """Flux principal du script : analyse les arguments et lance l'extraction."""
    ap = argparse.ArgumentParser(description="Dump d'un match LoL via l'API Riot")
    
    # Configuration des arguments CLI
    ap.add_argument("--api-key", type=str, help="Clé API Riot")
    ap.add_argument("--region", type=str, default="europe", help="europe/americas/asia")
    ap.add_argument("--platform", type=str, default="euw1", help="euw1/na1/kr")
    ap.add_argument("--name", type=str, help="Riot ID Name")
    ap.add_argument("--tag", type=str, help="Riot ID Tag")
    ap.add_argument("--queue", type=int, default=420, help="420=SoloQ, 440=Flex")
    ap.add_argument("--count", type=int, default=50, help="Nombre de matchs à scanner")
    ap.add_argument("--match-id", type=str, help="ID spécifique du match")
    
    args = ap.parse_args()

    # Gestion de la clé API (priorité à l'argument CLI)
    if args.api_key:
        os.environ["RIOT_API_KEY"] = args.api_key
    api = os.getenv("RIOT_API_KEY")
    if not api:
        raise SystemExit("RIOT_API_KEY manquante. Utilisez --api-key.")

    region = args.region.lower()
    platform = args.platform.lower()

    rw = RiotWatcher(api)
    lol = LolWatcher(api)

    # Détermination du match à traiter
    if args.match_id:
        match_id = args.match_id
        print(f"[INFO] match-id fourni: {match_id}")
    else:
        if not (args.name and args.tag):
            raise SystemExit("Il faut soit --match-id, soit --name et --tag.")
        puuid = get_puuid(rw, lol, region, platform, args.name, args.tag)
        match_id = pick_match_id(lol, region, puuid, queue=args.queue, count=args.count)

    # ---- Récupération des données ----
    print(f"[STEP] Téléchargement du match: {match_id}")
    match = lol.match.by_id(region, match_id)
    save_json(DATA_DIR / f"match_{match_id}.json", match)
    info = match.get("info", {})

    # ---- Export CSV : Participants ----
    df_p = flatten_participants(info)
    df_to_csv(DATA_DIR / f"participants_{match_id}.csv", df_p)
    print(f"[OK] participants CSV -> {DATA_DIR}")

    # ---- Export CSV : Équipes et Objectifs ----
    df_t = flatten_teams(info)
    df_to_csv(DATA_DIR / f"teams_{match_id}.csv", df_t)
    df_o = flatten_objectives_long(info)
    df_to_csv(DATA_DIR / f"objectives_{match_id}.csv", df_o)

    # ---- Récupération Timeline (Optionnelle) ----
    try:
        timeline = lol.match.timeline_by_match(region, match_id)
        save_json(DATA_DIR / f"timeline_{match_id}.json", timeline)
        df_ev = flatten_timeline_events(timeline)
        if not df_ev.empty:
            df_to_csv(DATA_DIR / f"timeline_events_{match_id}.csv", df_ev)
            print(f"[OK] timeline extraite.")
        time.sleep(1.0) # Respect du rate limit
    except ApiError as e:
        print(f"[WARN] Timeline indisponible: {e}")

    # Résumé final
    print("\n=== RÉSUMÉ ===")
    print(f"Match ID: {match_id}")
    print(f"Version: {info.get('gameVersion')}")
    print(f"Vainqueur: {[t.get('teamId') for t in info.get('teams', []) if t.get('win')]}")
    print("======== FIN ========")

if __name__ == "__main__":
    main()