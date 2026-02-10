"""
Tests for ingame_tab.py — utility functions (weighted_note, routing conversion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingame_tab import (
    account_routing_from_match_routing,
    normalize_riot_id,
    weighted_note,
)


# ──────────────────────────────────────────────────────────────────────────────
# weighted_note
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightedNote:
    def test_zero_total_games(self):
        assert weighted_note(60.0, 50.0, 5, 0, 20) == 0.0

    def test_returns_float(self):
        result = weighted_note(55.0, 50.0, 10, 50, 50)
        assert isinstance(result, float)

    def test_no_champ_games_returns_global(self):
        # When champ_games=0, confidence should be 0 → returns global_wr
        result = weighted_note(80.0, 50.0, 0, 50, 50)
        assert result == pytest.approx(50.0)

    def test_high_champ_games_weights_toward_base(self):
        # Many champion games should push toward the base (0.65*champ + 0.35*global)
        result = weighted_note(70.0, 50.0, 100, 100, 100)
        # base = 0.65*70 + 0.35*50 = 45.5 + 17.5 = 63.0
        # conf should be very high, so result ≈ 63.0
        assert result > 55.0

    def test_score_range(self):
        # Score should be in a reasonable range
        result = weighted_note(60.0, 50.0, 10, 50, 50)
        assert 0.0 <= result <= 100.0

    def test_negative_total_games(self):
        assert weighted_note(60.0, 50.0, 5, -1, 20) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# account_routing_from_match_routing
# ──────────────────────────────────────────────────────────────────────────────


class TestAccountRouting:
    def test_europe(self):
        assert account_routing_from_match_routing("europe") == "europe"

    def test_americas(self):
        assert account_routing_from_match_routing("americas") == "americas"

    def test_asia(self):
        assert account_routing_from_match_routing("asia") == "asia"

    def test_sea_fallback(self):
        assert account_routing_from_match_routing("sea") == "americas"

    def test_unknown_fallback(self):
        assert account_routing_from_match_routing("xyz") == "americas"


# ──────────────────────────────────────────────────────────────────────────────
# normalize_riot_id
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeRiotId:
    def test_basic(self):
        result = normalize_riot_id("player#EUW")
        assert result is not None
        name, tag = result
        assert name == "player"
        assert tag == "EUW"

    def test_with_spaces(self):
        result = normalize_riot_id("  player  # EUW  ")
        assert result is not None
        name, tag = result
        # gameName keeps internal spaces, tag is stripped
        assert "player" in name
        assert tag.strip() != ""

    def test_no_hash(self):
        result = normalize_riot_id("playerEUW")
        assert result is None

    def test_empty_string(self):
        result = normalize_riot_id("")
        assert result is None

    def test_only_hash(self):
        result = normalize_riot_id("#")
        # Either None or both parts are empty — either is acceptable
        if result is not None:
            name, tag = result
            assert name == "" or tag == ""
