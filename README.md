# SuperLOL Assistant

Projet de **Shutian LEI** et **Lixiang ZHANG**

Application desktop (PySide6) d'assistance pour League of Legends : tier-lists META, profil personnel, scan de partie en cours et hotkey pour envoi de messages en jeu.

Vous pouvez trouver la clé API dans sur le site suivant https://developer.riotgames.com/, en créant un compte bien sur. 

Les informations concernant le projet sphinx sont dans le fichier \build\html\index.html, il suffit de l'ouvrir dans le navigateur. 

---

## Architecture du projet

```
superLOLassistant-main/
├── interface.py            # Programme principal (GUI + logique)
├── ingame_tab.py           # Onglet "IN GAME" (scan LiveClient / Spectator)
├── trashbase.py            # Hotkey listener + envoi de texte en jeu (Windows API)
├── data_base_riot.py       # Collecte massive de matchs via Riot API (CSV)
├── lol_matchups_test.py    # Matchups lane-vs-lane (demo / Riot / recommandations)
├── riot_dump_one_match.py  # Dump complet d'un seul match (JSON + CSV)
├── data/                   # Cache champion tags, matchups, matches_raw
├── data_db/                # Sortie de data_base_riot (participants.csv, matches.csv)
├── data_user/              # Données personnelles collectees via le profil
└── data_one_match/         # Sortie de riot_dump_one_match
```

---

## Description des fichiers

### `interface.py` -- Programme principal

Point d'entree de l'application. Lance une fenetre PySide6 avec trois onglets :

- **META (global)** (`MetaTab`) : charge un CSV de participants, calcule des tier-lists (S/A/B/C) par role selon le winrate ou le KDA, et affiche les matchups au clic sur un champion.
- **MON PROFIL** (`ProfileTab`) : collecte les matchs d'un joueur via la Riot API ou charge un CSV personnel, affiche des stats par role et par champion, avec camembert interactif (drill-down par classe de champion).
- **IN GAME** (`InGameTab`, importe depuis `ingame_tab.py`) : scan de la partie en cours.

Fonctionnalites supplementaires :
- Basculement theme clair / sombre.
- Configuration d'un hotkey pour envoyer un resume de la partie en cours dans le chat du jeu (via `trashbase.py`).

### `ingame_tab.py` -- Onglet IN GAME

Widget PySide6 qui scanne la partie en cours et affiche des statistiques par joueur (allies / ennemis).

Deux sources de donnees :
1. **LiveClient** (`http://127.0.0.1:2999/liveclientdata/allgamedata`) -- disponible quand le joueur local est en jeu.
2. **Spectator API** (v5 via HTTP, fallback v4 via riotwatcher) -- quand LiveClient n'est pas disponible.

Pour chaque joueur, le module :
- Recupere les derniers matchs via `match-v5`.
- Calcule le winrate global, le winrate sur le champion, le KDA.
- Attribue une note ponderee (`weighted_note`) combinant confiance et performance.

### `trashbase.py` -- Hotkey et envoi de messages en jeu

Module Windows qui gere :
- **Hotkey listener** (`pynput`) : ecoute une touche configurable. Quand elle est pressee, appelle un callback et envoie le texte retourne dans le chat du jeu.
- **Envoi de texte** : utilise `ctypes` / Windows `SendInput` API pour simuler des frappes Unicode dans la fenetre de jeu (Enter pour ouvrir le chat, caracteres un par un, Enter pour envoyer).
- **Mode standalone** : peut aussi etre lance seul (`python trashbase.py`) pour envoyer des messages predsfinis ou aleatoires depuis `taunts.txt`.

### `data_base_riot.py` -- Collecte massive de matchs

Script CLI pour collecter N matchs LoL et produire deux CSV :
- `participants.csv` : une ligne par participant par match (matchId, teamId, role, championName, kills, deaths, assists, etc.)
- `matches.csv` : une ligne par match (matchId, winnerTeamId)

Methode de collecte :
- **Seed** : a partir de Riot IDs fournis (`--seed-riotids`) ou fallback automatique via le ladder (MASTER / GM / CHALLENGER).
- **Snowball** : les PUUIDs decouverts dans chaque match sont ajoutes a la file pour explorer de nouveaux matchs.

```bash
# Exemple
python data_base_riot.py --api-key RGAPI-XXXX --region europe --platform euw1 \
  --target 1000 --queue 420 --seed-riotids "player1#EUW,player2#EUW"
```

### `lol_matchups_test.py` -- Matchups et recommandations

Script CLI avec trois modes :

1. **`--demo`** : genere des matchs synthetiques (offline, sans cle API) et les sauvegarde en JSONL.
2. **`--riot`** : collecte de vrais matchs via la Riot API et sauvegarde en JSONL.
3. **`--recommend`** : recommande les meilleurs picks contre un champion ennemi donne, a partir du fichier `matchups.csv`.

Le flag `--build` (combinable avec `--demo` ou `--riot`) parse le JSONL brut et calcule les winrates lane-vs-lane pour produire `matchups.csv`.

```bash
# Demo offline
python lol_matchups_test.py --demo --build

# Recommandation
python lol_matchups_test.py --recommend --role mid --enemy Zed --topk 5 --min-games 20
```

### `riot_dump_one_match.py` -- Dump detaille d'un match

Script CLI qui telecharge et extrait les donnees completes d'un seul match :
- `match_{id}.json` : JSON brut du match.
- `participants_{id}.csv` : stats detaillees de chaque participant (items, runes, KDA, damage, vision, etc.).
- `teams_{id}.csv` : bans, objectifs par equipe.
- `objectives_{id}.csv` : objectifs en format long (baron, dragon, tower, etc.).
- `timeline_{id}.json` / `timeline_events_{id}.csv` : evenements de la timeline (kills, items, wards, etc.).

```bash
# Via Riot ID du joueur (match aleatoire parmi les recents)
python riot_dump_one_match.py --api-key RGAPI-XXXX --region europe --platform euw1 \
  --name ztheo17 --tag EUW --count 50

# Via match ID precis
python riot_dump_one_match.py --api-key RGAPI-XXXX --region europe --match-id EUW1_6999999999
```

---

## Dependances

```
pip install PySide6 riotwatcher pandas matplotlib requests numpy pynput pyperclip
```

| Package       | Usage                                              |
|---------------|----------------------------------------------------|
| PySide6       | Interface graphique Qt6                            |
| riotwatcher   | Client Python pour la Riot API                     |
| pandas        | Manipulation de DataFrames / CSV                   |
| matplotlib    | Graphiques (camemberts, etc.)                      |
| requests      | Requetes HTTP (Data Dragon, LiveClient, Spectator) |
| numpy         | Calculs numeriques                                 |
| pynput        | Ecoute clavier (hotkey)                            |
| pyperclip     | Copie dans le presse-papiers                       |

---

## Lancement

```bash
# Lancer l'application principale
python interface.py
```

---

## Cle API Riot

Une cle API Riot (`RGAPI-...`) est necessaire pour les fonctionnalites en ligne.
Elle peut etre fournie :
- Dans le champ "API Key" de l'interface.
- Via la variable d'environnement `RIOT_API_KEY`.
- Via l'argument `--api-key` pour les scripts CLI.
