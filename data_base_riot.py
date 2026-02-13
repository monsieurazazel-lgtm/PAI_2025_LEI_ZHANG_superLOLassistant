#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collecte N matchs LoL en CSV, en seedant avec des Riot IDs (gameName#tagLine).
"""

from __future__ import annotations
import argparse
import os
import time
import random
import collections
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Deque, Tuple
import pandas as pd

# --------- Riot deps ----------
try:
    from riotwatcher import LolWatcher, RiotWatcher, ApiError
except Exception as e:
    raise SystemExit("Installe: pip install riotwatcher pandas\n" + str(e))

# --------- Rôles ----------
ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "UTILITY": "sup",
}

# --------- Rate limit ----------
SLEEP_PER_CALL = 1.3  # ≈92 req / 2 min
BACKOFF_429 = 3.0


def sleep_brief() -> None:
    """Effectue une courte pause pour respecter le rate limit de l'API."""
    time.sleep(SLEEP_PER_CALL)


def safe_call(fn, *args, **kwargs) -> Any:
    """Exécute un appel API de manière sécurisée avec gestion des erreurs courantes.

    Args:
        fn: La fonction de l'API Riot à appeler.
        *args: Arguments positionnels pour la fonction.
        **kwargs: Arguments nommés pour la fonction.

    Returns:
        Le résultat de l'appel API.

    Raises:
        SystemExit: Si la clé API est invalide ou expirée.
        Exception: Pour toute autre erreur API non gérée.
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
                raise SystemExit(
                    "Clé API invalide/expirée (401/403). Mets RIOT_API_KEY à jour."
                )
            raise


# --------- Extraction ----------
def extract_winner_team_id(info: Dict[str, Any]) -> int | None:
    """Extrait l'ID de l'équipe gagnante depuis les informations du match.

    Args:
        info: Dictionnaire 'info' provenant de l'objet match de l'API.

    Returns:
        L'ID de l'équipe victorieuse ou None si non trouvé.
    """
    wins = [t.get("teamId") for t in (info.get("teams") or []) if t.get("win")]
    return wins[0] if wins else None


def iter_participant_rows(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Génère une liste de dictionnaires représentant chaque participant du match.

    Args:
        match: Objet match complet renvoyé par l'API Match-v5.

    Returns:
        Liste de lignes de données formatées pour le CSV participants.
    """
    meta = match.get("metadata", {})
    info = match.get("info", {})
    match_id = meta.get("matchId")
    winner_team = extract_winner_team_id(info)
    out = []
    for p in info.get("participants", []):
        role = ROLE_MAP.get((p.get("teamPosition") or "").upper())
        if not role:
            continue
        k = int(p.get("kills", 0))
        d = int(p.get("deaths", 0))
        a = int(p.get("assists", 0))
        kda = (k + a) / (d if d > 0 else 1)
        out.append(
            {
                "matchId": match_id,
                "teamId": p.get("teamId"),
                "teamWin": bool(p.get("win")),
                "winnerTeamId": winner_team,
                "role": role,
                "championName": p.get("championName"),
                "kills": k,
                "deaths": d,
                "assists": a,
                "kda_ratio": float(f"{kda:.3f}"),
                "summoner1Id": p.get("summoner1Id"),
                "summoner2Id": p.get("summoner2Id"),
                "puuid": p.get("puuid"),
            }
        )
    return out


def rows_schema() -> List[str]:
    """Définit l'ordre et le nom des colonnes pour le fichier participants.csv.

    Returns:
        Liste des noms de colonnes.
    """
    return [
        "matchId", "teamId", "teamWin", "winnerTeamId", "role",
        "championName", "kills", "deaths", "assists", "kda_ratio",
        "summoner1Id", "summoner2Id", "puuid",
    ]


# --------- IO ----------
def save_append_csv(path: Path, rows: List[Dict[str, Any]], header: bool) -> None:
    """Sauvegarde ou ajoute des lignes au fichier CSV des participants.

    Args:
        path: Chemin du fichier CSV.
        rows: Liste des dictionnaires de données à sauvegarder.
        header: Si True, crée le fichier avec l'en-tête. Si False, ajoute sans en-tête.
    """
    if not rows and not header:
        return
    df = pd.DataFrame(rows, columns=rows_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode=("w" if header else "a"), index=False, header=header)


def save_matches_csv(
    path: Path, match_rows: List[Tuple[str, int | None]], header: bool
) -> None:
    """Sauvegarde ou ajoute des lignes au fichier CSV récapitulatif des matchs.

    Args:
        path: Chemin du fichier CSV.
        match_rows: Liste de tuples (matchId, winnerTeamId).
        header: Si True, écrit l'en-tête du fichier.
    """
    if not match_rows and not header:
        return
    df = pd.DataFrame(match_rows, columns=["matchId", "winnerTeamId"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode=("w" if header else "a"), index=False, header=header)


# --------- Seed helpers ----------
def normalize_riot_id(s: str) -> tuple[str, str] | None:
    """Normalise une chaîne Riot ID au format gameName#tagLine.

    Args:
        s: Chaîne brute (ex: 'Nom # TAG').

    Returns:
        Tuple (gameName, tagLine) normalisé ou None si le format est invalide.
    """
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    if "#" not in s:
        return None
    game, tag = s.split("#", 1)
    if not game or not tag:
        return None
    return game, tag.upper()


def riotids_to_puuids(
    rw: RiotWatcher, region_routing: str, riotids: list[str]
) -> list[str]:
    """Convertit une liste de Riot IDs en PUUIDs via l'API Account-v1.

    Args:
        rw: Instance du RiotWatcher.
        region_routing: Région géographique (europe, americas, etc.).
        riotids: Liste de chaînes 'gameName#tagLine'.

    Returns:
        Liste des PUUIDs correspondants trouvés.
    """
    out = []
    for rid in riotids:
        tup = normalize_riot_id(rid)
        if not tup:
            continue
        game, tag = tup
        try:
            acc = safe_call(rw.account.by_riot_id, region_routing, game, tag)
            p = acc.get("puuid")
            if p:
                out.append(p)
        except ApiError:
            continue
    return list({p for p in out if p})


def league_entries_pages(
    lol: LolWatcher, platform_lc: str, queue_str: str, tier_en: str, div: str, max_pages: int = 10
) -> List[dict]:
    """Récupère plusieurs pages d'entrées d'une ligue donnée.

    Args:
        lol: Instance du LolWatcher.
        platform_lc: Identifiant plateforme en minuscules (ex: 'euw1').
        queue_str: Type de file (ex: 'RANKED_SOLO_5x5').
        tier_en: Tier (ex: 'DIAMOND').
        div: Division (ex: 'I').
        max_pages: Nombre maximum de pages à parcourir.

    Returns:
        Liste des entrées de ligue collectées.
    """
    all_entries = []
    for pg in range(1, max_pages + 1):
        entries = []
        try:
            entries = safe_call(
                lol.league.entries, platform_lc, queue_str, tier_en, div, page=pg
            )
        except ApiError:
            entries = []
        if not entries:
            try:
                entries = safe_call(
                    lol.league.entries, platform_lc, tier_en, div, queue_str, page=pg
                )
            except ApiError:
                entries = []
        if not entries:
            break
        all_entries.extend(entries)
    return all_entries


def seed_from_ladder_hightiers(
    lol: LolWatcher, platform_lc: str, queue_str: str
) -> List[str]:
    """Récupère des summonerIds depuis le haut du ladder pour amorcer la collecte.

    Args:
        lol: Instance du LolWatcher.
        platform_lc: Identifiant plateforme.
        queue_str: Type de file.

    Returns:
        Liste unique de summonerIds.
    """
    ids: List[str] = []
    # Parcours successif MASTER -> GM -> CHALL
    for method in [lol.league.masters_by_queue, lol.league.grandmaster_by_queue, lol.league.challenger_by_queue]:
        try:
            data = safe_call(method, platform_lc, queue_str)
            ids += [e.get("summonerId") for e in (data.get("entries") or []) if e.get("summonerId")]
            if ids: break
        except ApiError:
            continue
    
    # Fallback DIAMOND
    if not ids:
        for div in ["I", "II", "III", "IV"]:
            entries = league_entries_pages(lol, platform_lc, queue_str, "DIAMOND", div, max_pages=5)
            ids += [e.get("summonerId") for e in entries if e.get("summonerId")]
            if ids: break

    return list({x for x in ids if x})


def summoner_ids_to_puuids(
    lol: LolWatcher, platform_lc: str, summ_ids: List[str]
) -> List[str]:
    """Convertit une liste de summonerIds en PUUIDs.

    Args:
        lol: Instance du LolWatcher.
        platform_lc: Identifiant plateforme.
        summ_ids: Liste des summonerIds.

    Returns:
        Liste des PUUIDs correspondants.
    """
    puuids = []
    for sid in summ_ids:
        try:
            s = safe_call(lol.summoner.by_id, platform_lc, sid)
            if s.get("puuid"):
                puuids.append(s["puuid"])
        except ApiError:
            pass
    return list({p for p in puuids if p})


# --------- Collecte ----------
def collect_dataset(
    api_key: str,
    region: str,
    platform: str,
    target_matches: int,
    queue_id: int | None,
    outdir: Path,
    matchlist_count: int = 100,
    max_seed_players: int = 300,
    seed_riotids: List[str] | None = None,
) -> None:
    """Exécute la boucle principale de collecte de données par propagation (snowball).

    Args:
        api_key: Clé API Riot Games.
        region: Région routing (europe, americas, etc.).
        platform: Shard plateforme (euw1, na1, etc.).
        target_matches: Nombre de matchs total à collecter.
        queue_id: ID de la file (420, 440) ou None.
        outdir: Dossier de sortie.
        matchlist_count: Nombre de matchs à récupérer par joueur.
        max_seed_players: Nombre max de joueurs initiaux.
        seed_riotids: Liste optionnelle de Riot IDs pour amorcer.
    """
    rw = RiotWatcher(api_key)
    lol = LolWatcher(api_key)

    platform_lc = platform.lower().strip()
    region_lc = region.lower().strip()
    QUEUE_STR = "RANKED_SOLO_5x5" if (queue_id == 420 or queue_id is None) else "RANKED_FLEX_SR"

    outdir.mkdir(parents=True, exist_ok=True)
    part_csv = outdir / "participants.csv"
    match_csv = outdir / "matches.csv"
    save_append_csv(part_csv, [], header=True)
    save_matches_csv(match_csv, [], header=True)

    # 1) Amorsage (Seeds)
    seeds_puuids: List[str] = []
    if seed_riotids:
        seeds_puuids = riotids_to_puuids(rw, region_lc, seed_riotids)

    if not seeds_puuids:
        summ_ids = seed_from_ladder_hightiers(lol, platform_lc, QUEUE_STR)
        if max_seed_players and len(summ_ids) > max_seed_players:
            random.shuffle(summ_ids)
            summ_ids = summ_ids[:max_seed_players]
        seeds_puuids = summoner_ids_to_puuids(lol, platform_lc, summ_ids)

    if not seeds_puuids:
        raise SystemExit("Aucun PUUID seed disponible.")

    # 2) Parcours
    puuid_queue: Deque[str] = collections.deque(seeds_puuids)
    seen_puuids: Set[str] = set(seeds_puuids)
    seen_matches: Set[str] = set()

    processed = 0
    batch_rows = []
    batch_match_rows = []

    print(f"[RUN] cible={target_matches} matchs, queue_id={queue_id}, seeds={len(seeds_puuids)}")

    while processed < target_matches and puuid_queue:
        puuid = puuid_queue.popleft()
        kw = {"queue": queue_id, "type": "ranked"} if queue_id else {}
        
        try:
            mlist = safe_call(lol.match.matchlist_by_puuid, region_lc, puuid, count=matchlist_count, **kw)
        except ApiError:
            continue
        
        if not mlist: continue

        for mid in mlist:
            if processed >= target_matches: break
            if mid in seen_matches: continue
            
            try:
                match = safe_call(lol.match.by_id, region_lc, mid)
                info = match.get("info", {})
                if not info or not info.get("participants"): continue
                
                p_rows = iter_participant_rows(match)
                if not p_rows: continue

                winner_team = extract_winner_team_id(info)
                batch_rows.extend(p_rows)
                batch_match_rows.append((mid, winner_team))
                seen_matches.add(mid)
                processed += 1

                # Snowball : ajout des nouveaux joueurs rencontrés
                for pr in p_rows:
                    pu = pr["puuid"]
                    if pu and pu not in seen_puuids:
                        seen_puuids.add(pu)
                        puuid_queue.append(pu)

                if len(batch_rows) >= 500:
                    save_append_csv(part_csv, batch_rows, header=False)
                    save_matches_csv(match_csv, batch_match_rows, header=False)
                    print(f"[SAVE] {processed}/{target_matches} matchs")
                    batch_rows.clear()
                    batch_match_rows.clear()
            except ApiError:
                continue

    # Final Flush
    if batch_rows:
        save_append_csv(part_csv, batch_rows, header=False)
        save_matches_csv(match_csv, batch_match_rows, header=False)

    print(f"[DONE] Collecte terminée : {processed} matchs.")


def main() -> None:
    """Parse les arguments CLI et lance la collecte."""
    ap = argparse.ArgumentParser(description="Collecte N matchs (seed via Riot IDs)")
    ap.add_argument("--api-key", type=str, help="Clé Riot")
    ap.add_argument("--region", type=str, default="europe", help="europe/americas/asia/sea")
    ap.add_argument("--platform", type=str, default="euw1", help="euw1/na1/kr/...")
    ap.add_argument("--target", type=int, default=1000, help="Nombre de matchs")
    ap.add_argument("--queue", type=int, default=420, help="420=SoloQ, 440=Flex")
    ap.add_argument("--matchlist-count", type=int, default=100, help="Max 100")
    ap.add_argument("--outdir", type=str, default="data_db", help="Dossier de sortie")
    ap.add_argument("--max-seed-players", type=int, default=300, help="Limite seeds")
    ap.add_argument("--seed-riotids", type=str, help="IDs séparés par des virgules")
    ap.add_argument("--seed-riotids-file", type=str, help="Fichier texte, un ID par ligne")

    args = ap.parse_args()

    if args.api_key: os.environ["RIOT_API_KEY"] = args.api_key
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        raise SystemExit("RIOT_API_KEY absente.")

    riotids = []
    if args.seed_riotids:
        riotids += [s.strip() for s in args.seed_riotids.split(",") if s.strip()]
    if args.seed_riotids_file and Path(args.seed_riotids_file).exists():
        with open(args.seed_riotids_file, "r", encoding="utf-8") as f:
            riotids += [ln.strip() for ln in f if ln.strip()]

    collect_dataset(
        api_key=api_key,
        region=args.region.lower().strip(),
        platform=args.platform.lower().strip(),
        target_matches=args.target,
        queue_id=(args.queue if args.queue != 0 else None),
        outdir=Path(args.outdir),
        matchlist_count=max(1, min(100, args.matchlist_count)),
        max_seed_players=max(50, args.max_seed_players),
        seed_riotids=(riotids or None),
    )


if __name__ == "__main__":
    main()