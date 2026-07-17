import unittest
from unittest.mock import patch

import util.mlb_stats_data_util as stats
import test_data as data

class TestMlbStatsData(unittest.TestCase):

  @patch('util.data_util.get_league_batting_averages')
  def test_get_mlb_batting_national_averages(self, mock_get_league_batting_averages):

    # Arrange
    mock_get_league_batting_averages.return_value = data.get_league_batting_averages()

    # Act
    actual = stats.get_mlb_batting_national_averages()

    assert len(actual) > 0

  @patch('util.data_util.get_pitching_stats')
  @patch('util.data_util.get_statcast_pitching_stats')
  @patch('util.data_util.get_statcast_data')
  def test_get_mlb_pitcher_profile(
    self, mock_get_statcast_data, mock_get_statcast_pitching_stats, mock_get_pitching_stats):

    # Arrange
    mock_get_statcast_data.return_value = data.get_statcast_dataset()
    mock_get_statcast_pitching_stats.return_value = data.get_statcast_pitching_stats()
    mock_get_pitching_stats.return_value = data.get_pitcher_stats()

    # Act
    actual = stats.get_mlb_pitcher_profile()

    assert len(actual) == 7

  @patch('util.data_util.get_batting_stats')
  @patch('util.data_util.get_statcast_data')
  def test_get_mlb_batter_profile(
    self, mock_get_statcast_data, mock_get_batting_stats):

    # Arrange
    mock_get_statcast_data.return_value = data.get_statcast_dataset()
    mock_get_batting_stats.return_value = data.get_batting_stats()

    # Act
    actual = stats.get_mlb_batting_profile()

    assert len(actual) == 36

  def test_get_mlb_live_stats(self):

    # Arrange
    target_date = "2026-07-22"

    # Act
    actual = stats.fetch_mlb_live_stats(target_date)

    # Assert
    assert len(actual) > 0
