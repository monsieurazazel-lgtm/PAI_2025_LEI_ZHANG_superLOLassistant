#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lol_matchups_test.py
Analyse des duels (matchups) League of Legends par rôle.
Permet la génération de données démo, la collecte via API Riot et la recommandation de champions.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# ===============================
#          CONSTANTES
# ===============================
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "matches_raw.jsonl"
MATCHUPS_CSV = DATA_DIR / "matchups.csv"

# Mapping des rôles de l'API Riot vers un format simplifié
ROLE_MAP: Dict[str, str] = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "UTILITY": "sup",
}

# Liste de champions pour le mode démo
DEMO_CHAMPS: List[str] = [
    "Ahri", "Zed", "Yone", "Orianna", "Annie", "Garen", "Darius", "Jax",
    "Camille", "Riven", "LeeSin", "Vi", "Sejuani", "Kayn", "Graves",
    "Jinx", "Caitlyn", "Ashe", "Xayah", "Ezreal", "Thresh", "Lulu",
    "Leona", "Nautilus", "Morgana",
]
DEMO_ROLES: List[str] = ["top", "jungle", "mid", "bot", "sup"]


# ===============================
#           MODE DÉMO
# ===============================
def demo_generate_matches(n_matches: int = 200) -> pd.DataFrame:
    """Génère des matchs synthétiques (5 rôles x 2 équipes) pour tester la chaîne.

    Args:
        n_matches: Nombre de matchs à générer. Par défaut 200.

    Returns:
        pd.DataFrame: Tableau contenant les données de matchs générées.
    """
    rows = []
    for m in range(n_matches):
        match_id = f"DEMO_{m:06d}"
        # Sélection aléatoire des champions pour les alliés et ennemis
        ally = {r: random.choice(DEMO_CHAMPS) for r in DEMO_ROLES}
        enemy = {
            r: random.choice([c for c in DEMO_CHAMPS if c != ally[r]] or DEMO_CHAMPS)
            for r in DEMO_ROLES
        }

        # Simulation d'un biais de victoire pour Jinx/Thresh
        bias = 0.0
        if ally["bot"] == "Jinx" and ally["sup"] == "Thresh":
            bias += 0.02
        if enemy["bot"] == "Jinx" and enemy["sup"] == "Thresh":
            bias -= 0.02
        ally_win = random.random() < (0.50 + bias)

        # Ajout des données par rôle et par équipe
        for r in DEMO_ROLES:
            rows.append({
                "matchId": match_id, "teamId": 100, "win": ally_win,
                "role": r, "champ": ally[r],
            })
            rows.append({
                "matchId": match_id, "teamId": 200, "win": (not ally_win),
                "role": r, "champ": enemy[r],
            })
    return pd.DataFrame(rows)


def save_raw_from_df(df: pd.DataFrame) -> None:
    """Écrit un fichier JSONL brut simulant le format de l'API Riot.

    Args:
        df: Le DataFrame contenant les données de matchs à sauvegarder.
    """
    DATA_DIR.mkdir(exist_ok=True)
    inv = {v: k for k, v in ROLE_MAP.items()}
    with RAW_PATH.open("w", encoding="utf-8") as f:
        for mid, sub in df.groupby("matchId"):
            parts = []
            for _, row in sub.iterrows():
                parts.append({
                    "teamId": int(row["teamId"]),
                    "win": bool(row["win"]),
                    "teamPosition": inv.get(row["role"], "MIDDLE"),
                    "championName": row["champ"],
                })
            m = {
                "metadata": {"matchId": mid},
                "info": {"participants": parts, "gameVersion": "DEMO-1.0"},
            }
            f.write(json.dumps(m) + "\n")


# ===============================
#          MODE API RIOT
# ===============================
def riot_collect(
    api_key: str, platform: str, region: str, game_name: str, tag_line: str,
    queue: int = 440, count: int = 200, pause_sec: float = 1.2,
) -> None:
    """Collecte des matchs réels via l'API Riot Games.

    Récupère le PUUID, la liste des matchs, puis télécharge chaque match dans un fichier JSONL.

    Args:
        api_key: Clé API Riot.
        platform: Plateforme (ex: EUW1).
        region: Région de routage (ex: europe).
        game_name: Nom de jeu Riot.
        tag_line: Tag Riot.
        queue: ID de la file (440=Flex).
        count: Nombre de matchs à collecter.
        pause_sec: Temps de pause entre les appels API pour éviter le 'Rate Limit'.
    """
    print("[RIOT] Import des clients Riot…")
    try:
        from riotwatcher import RiotWatcher, LolWatcher, ApiError
    except ImportError as e:
        raise SystemExit("riotwatcher n'est pas installé. Fais: pip install riotwatcher\n" + str(e))

    rw = RiotWatcher(api_key)
    lol = LolWatcher(api_key)
    DATA_DIR.mkdir(exist_ok=True)

    # 1) Récupération du PUUID
    print(f"[RIOT] Recherche du PUUID pour {game_name}#{tag_line}")
    try:
        acct = rw.account.by_riot_id(region, game_name, tag_line)
        puuid = acct.get("puuid")
        if not puuid: raise ValueError("PUUID absent.")
    except Exception as e:
        print(f"[RIOT] Erreur account-v1, tentative fallback summoner-v4... ({e})")
        summ = lol.summoner.by_name(platform, game_name)
        puuid = summ.get("puuid")

    # 2) Liste des matchs
    match_ids = lol.match.matchlist_by_puuid(region, puuid, type="ranked", queue=queue, count=count)
    print(f"[RIOT] {len(match_ids)} matchs trouvés.")

    # Dé-duplication
    seen = set()
    if RAW_PATH.exists():
        with RAW_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                    mid = m.get("metadata", {}).get("matchId")
                    if mid: seen.add(mid)
                except: pass

    # 3) Téléchargement
    fetched = 0
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for i, mid in enumerate(match_ids, 1):
            if mid in seen: continue
            try:
                mat = lol.match.by_id(region, mid)
                f.write(json.dumps(mat) + "\n")
                fetched += 1
                time.sleep(pause_sec)
            except ApiError as e:
                if e.response.status_code == 429:
                    print("[RIOT] 429 Rate Limit → Pause 3s")
                    time.sleep(3.0)
    print(f"[RIOT] Terminé. {fetched} nouveaux matchs ajoutés.")


# ===============================
#       PARSING & MATCHUPS
# ===============================
def flatten_matches(jsonl_path: Path) -> pd.DataFrame:
    """Transforme les données JSONL brutes en un DataFrame exploitable.

    Args:
        jsonl_path: Chemin du fichier JSONL.

    Returns:
        pd.DataFrame: Données nettoyées (matchId, teamId, win, role, champ).
    """
    rows = []
    if not jsonl_path.exists():
        return pd.DataFrame(columns=["matchId", "teamId", "win", "role", "champ"])

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
                info = m.get("info", {})
                for p in info.get("participants", []):
                    role = ROLE_MAP.get((p.get("teamPosition") or "").upper())
                    if not role: continue
                    rows.append({
                        "matchId": m.get("metadata", {}).get("matchId"),
                        "teamId": p.get("teamId"),
                        "win": bool(p.get("win")),
                        "role": role,
                        "champ": p.get("championName"),
                    })
            except: continue

    df = pd.DataFrame(rows)
    # On ne garde que les matchs complets (10 participants)
    if not df.empty:
        valid = df.groupby("matchId").size().eq(10)
        df = df[df["matchId"].isin(valid[valid].index)]
    return df


def compute_lane_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques de duel (winrate) par voie.

    Args:
        df: DataFrame plat des matchs.

    Returns:
        pd.DataFrame: Statistiques de matchups (victoires, jeux, winrate).
    """
    if df.empty:
        return pd.DataFrame(columns=["role", "champ_ally", "champ_enemy", "games", "wins", "winrate"])

    # Fusion des données alliées (100) et ennemies (200) pour créer les duels
    left = df[df.teamId == 100].groupby(["matchId", "role"]).first().reset_index()
    right = df[df.teamId == 200].groupby(["matchId", "role"]).first().reset_index()
    duel = left.merge(right, on=["matchId", "role"], suffixes=("_ally", "_enemy"))

    duel["ally_win"] = duel["win_ally"].astype(int)
    grp = duel.groupby(["role", "champ_ally", "champ_enemy"]).agg(
        games=("ally_win", "size"),
        wins=("ally_win", "sum"),
    ).reset_index()

    grp["winrate"] = grp["wins"] / grp["games"].replace(0, 1)
    return grp.sort_values(["role", "champ_ally", "games"], ascending=[True, True, False])


def save_matchups_csv(df: pd.DataFrame) -> None:
    """Sauvegarde les résultats des matchups dans un fichier CSV.

    Args:
        df: Le DataFrame des matchups à sauvegarder.
    """
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(MATCHUPS_CSV, index=False)
    print(f"[BUILD] matchups.csv écrit dans {MATCHUPS_CSV}")


def recommend(role: str, enemy: str, topk: int = 5, min_games: int = 20) -> pd.DataFrame:
    """Recommande les meilleurs champions à choisir contre un ennemi donné.

    Args:
        role: Le rôle visé.
        enemy: Le champion ennemi à contrer.
        topk: Nombre de recommandations à retourner.
        min_games: Nombre minimal de parties pour être statistique.

    Returns:
        pd.DataFrame: Les top recommandations triées par winrate.
    """
    if not MATCHUPS_CSV.exists():
        raise SystemExit("matchups.csv introuvable. Lancez d'abord --build.")
    
    m = pd.read_csv(MATCHUPS_CSV)
    sub = m[(m["role"] == role) & (m["champ_enemy"] == enemy) & (m["games"] >= min_games)]
    return sub.sort_values("winrate", ascending=False).head(topk)


# ===============================
#              CLI
# ===============================
def build_argparser() -> argparse.ArgumentParser:
    """Configure le gestionnaire d'arguments en ligne de commande."""
    p = argparse.ArgumentParser(description="Matchups LoL - Analyse et Recommandation")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Données synthétiques")
    mode.add_argument("--riot", action="store_true", help="Collecte API Riot")
    mode.add_argument("--recommend", action="store_true", help="Recommandation")

    p.add_argument("--api-key", type=str, help="Clé API Riot")
    p.add_argument("--platform", type=str, default="EUW1", help="ex: EUW1")
    p.add_argument("--region", type=str, default="europe", help="ex: europe")
    p.add_argument("--name", type=str, help="Nom Riot ID")
    p.add_argument("--tag", type=str, help="Tag Riot ID")
    p.add_argument("--build", action="store_true", help="Calculer matchups.csv")
    p.add_argument("--role", type=str, default="mid", help="Rôle ciblé")
    p.add_argument("--enemy", type=str, default="Zed", help="Champion ennemi")
    return p


def main() -> None:
    """Point d'entrée principal du programme."""
    args = build_argparser().parse_args()

    if args.api_key: os.environ["RIOT_API_KEY"] = args.api_key
    api_key = os.getenv("RIOT_API_KEY")

    # Mode DÉMO
    if args.demo:
        df_demo = demo_generate_matches(200)
        save_raw_from_df(df_demo)
        if args.build:
            df = flatten_matches(RAW_PATH)
            save_matchups_csv(compute_lane_matchups(df))
        return

    # Mode RIOT
    if args.riot:
        if not api_key: raise SystemExit("Clé API manquante.")
        riot_collect(api_key, args.platform, args.region, args.name, args.tag)
        if args.build:
            df = flatten_matches(RAW_PATH)
            save_matchups_csv(compute_lane_matchups(df))
        return

    # Mode RECOMMANDATION
    if args.recommend:
        rec = recommend(args.role, args.enemy)
        if rec.empty: print("Pas assez de données.")
        else: print(rec.to_string(index=False))


if __name__ == "__main__":
    main()