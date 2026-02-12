"""
Tests for interface.py — utility helpers, player stats, tier classification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# trashbase uses ctypes.windll (Windows-only); mock it so interface.py
# can be imported on macOS / Linux CI runners.
import unittest.mock as _mock

_windll_patch = _mock.MagicMock()
sys.modules.setdefault("ctypes.windll", _windll_patch)

import ctypes as _ctypes  # noqa: E402

if not hasattr(_ctypes, "windll"):
    _ctypes.windll = _windll_patch  # type: ignore[attr-defined]

from interface import (  # noqa: E402
    _df_is_empty,
    _mask_any,
    _series_is_empty,
    compute_player_by_champ,
    compute_player_by_role,
    compute_player_overview,
)


# ──────────────────────────────────────────────────────────────────────────────
# _df_is_empty
# ──────────────────────────────────────────────────────────────────────────────


class TestDfIsEmpty:
    def test_empty_dataframe(self):
        assert _df_is_empty(pd.DataFrame()) is True

    def test_non_empty_dataframe(self):
        assert _df_is_empty(pd.DataFrame({"a": [1]})) is False


# ──────────────────────────────────────────────────────────────────────────────
# _series_is_empty
# ──────────────────────────────────────────────────────────────────────────────


class TestSeriesIsEmpty:
    def test_empty_series(self):
        assert _series_is_empty(pd.Series(dtype=float)) is True

    def test_non_empty_series(self):
        assert _series_is_empty(pd.Series([1, 2, 3])) is False


# ──────────────────────────────────────────────────────────────────────────────
# _mask_any
# ──────────────────────────────────────────────────────────────────────────────


class TestMaskAny:
    def test_bool_true(self):
        assert _mask_any(True) is True

    def test_bool_false(self):
        assert _mask_any(False) is False

    def test_series_all_false(self):
        assert _mask_any(pd.Series([False, False, False])) is False

    def test_series_some_true(self):
        assert _mask_any(pd.Series([False, True, False])) is True

    def test_numpy_array(self):
        assert _mask_any(np.array([False, False, True])) is True

    def test_numpy_all_false(self):
        assert _mask_any(np.array([False, False])) is False

    def test_series_with_nan(self):
        # NaN should be treated as False
        assert _mask_any(pd.Series([np.nan, np.nan])) is False

    def test_series_with_nan_and_true(self):
        assert _mask_any(pd.Series([np.nan, True])) is True


# ──────────────────────────────────────────────────────────────────────────────
# compute_player_overview
# ──────────────────────────────────────────────────────────────────────────────


def _make_user_df(n_games=10, win_frac=0.6) -> DataFrame:
    """Helper: create a user participants DataFrame."""
    rows = []
    for i in range(n_games):
        rows.append(
            {
                "teamWin": i < int(n_games * win_frac),
                "kills": 5,
                "deaths": 3,
                "assists": 7,
                "role": "mid" if i % 2 == 0 else "top",
                "championName": "Ahri" if i % 3 != 2 else "Zed",
            }
        )
    return pd.DataFrame(rows)


class TestComputePlayerOverview:
    def test_empty_df(self):
        result = compute_player_overview(pd.DataFrame())
        assert result["games"] == 0
        assert result["wins"] == 0
        assert result["wr"] == 0.0
        assert result["kda"] == 0.0

    def test_basic_stats(self):
        df = _make_user_df(n_games=10, win_frac=0.6)
        result = compute_player_overview(df)
        assert result["games"] == 10
        assert result["wins"] == 6
        assert result["wr"] == pytest.approx(60.0)

    def test_kda_calculation(self):
        df = _make_user_df(n_games=1)
        result = compute_player_overview(df)
        # (5 + 7) / max(1, 3) = 4.0
        assert result["kda"] == pytest.approx(4.0)

    def test_kda_zero_deaths(self):
        df = pd.DataFrame([{"teamWin": True, "kills": 5, "deaths": 0, "assists": 7}])
        result = compute_player_overview(df)
        # (5 + 7) / max(1, 0) = 12.0
        assert result["kda"] == pytest.approx(12.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_player_by_role
# ──────────────────────────────────────────────────────────────────────────────


class TestComputePlayerByRole:
    def test_empty_df(self):
        # Even if the input DataFrame is empty, the output should still have the
        # expected structure (columns, etc.) but no rows.
        result = compute_player_by_role(pd.DataFrame())
        assert result.empty
        assert "role" in result.columns

    def test_groups_by_role(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_role(df)
        assert set(result["role"].unique()) == {"mid", "top"}

    def test_columns_present(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_role(df)
        for col in ("role", "games", "wins", "wr%", "kda", "K", "D", "A"):
            assert col in result.columns

    def test_games_sum_matches_total(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_role(df)
        assert result["games"].sum() == 10


# ──────────────────────────────────────────────────────────────────────────────
# compute_player_by_champ
# ──────────────────────────────────────────────────────────────────────────────


class TestComputePlayerByChamp:
    def test_empty_df(self):
        result = compute_player_by_champ(pd.DataFrame())
        assert result.empty
        assert "champion" in result.columns

    def test_groups_by_champion(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_champ(df)
        assert "Ahri" in result["champion"].values

    def test_columns_present(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_champ(df)
        for col in ("champion", "games", "wins", "wr%", "kda", "K", "D", "A"):
            assert col in result.columns

    def test_games_sum_matches_total(self):
        df = _make_user_df(n_games=10)
        result = compute_player_by_champ(df)
        assert result["games"].sum() == 10
