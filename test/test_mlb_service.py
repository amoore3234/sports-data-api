import unittest
from unittest.mock import patch
import services.mlb_service as service
import test_data as data
import util.data_util as mlb_data

class TestMlbService(unittest.TestCase):

  @patch('services.mlb_service.get_teams')
  @patch('services.mlb_service.get_list_of_starters')
  def test_generate_starters_pitchers(self, mock_get_list_of_starters, mock_get_teams):

    # Arrange
    pitcher_starters_df = data.get_pitcher_profile_data()
    mock_get_list_of_starters.return_value = data.get_starting_players_data()['Starting Lineup']
    mock_get_teams.return_value = ['STL', 'TBR', 'DET', 'CWS', 'MIL', 'CHC', 'KCR']

    # Act
    actual = service.generate_starters('pitchers', pitcher_starters_df)

    #Assert
    assert len(actual) > 0
    assert len(actual['name']) >= 7
    assert (actual['batting_order'] == 0).any()

  @patch('services.mlb_service.get_teams')
  @patch('services.mlb_service.get_list_of_starters')
  def test_generate_starters_hitters(self, mock_get_list_of_starters, mock_get_teams):

    # Arrange
    batter_starters_df = data.get_batter_profile_data()
    mock_get_list_of_starters.return_value = data.get_starting_players_data()['Starting Lineup']
    mock_get_teams.return_value = [
      'STL',
      'STL',
      'STL',
      'STL',
      'STL',
      'STL',
      'STL',
      'STL',
      'STL',
      'DET',
      'DET',
      'DET',
      'DET',
      'DET',
      'DET',
      'DET',
      'DET',
      'DET',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'CWS',
      'KCR',
      'KCR',
      'KCR',
      'KCR',
      'KCR',
      'KCR',
      'KCR',
      'KCR',
      'KCR'
    ]

    # Act
    actual = service.generate_starters('hitters', batter_starters_df)

    #Assert
    assert len(actual) > 0
    assert (actual['batting_order'] >= 1).any() == True
    assert len(actual['name']) >= 35

  def test_get_vegas_odds(self):
    # Arrange
    starting = data.get_starting_players_data()

    # Act
    odds = service.get_vegas_odds(starting)

    # Assert
    assert 'favorite_to_win' in odds.columns
    assert 'favorite_odds' in odds.columns
    assert 'over_and_under' in odds.columns

  def test_calculate_total_splits(self):
    # Arrange
    df = data.get_batter_profile_data().copy()
    df['favorite_odds'] = -150
    df['game_implied_total'] = 8.0

    # Act
    out = service.calculate_total_splits(df)

    # Assert
    assert 'team_implied_total' in out.columns
    assert 'opposing_team_implied_total' in out.columns

  def test_get_pitcher_data_for_model_and_process(self):
    # Arrange
    pitchers = data.get_pitcher_profile_data()
    pitchers['team'] = ['STL', 'TBR', 'DET', 'CWS', 'MIL', 'CHC', 'KCR']
    pitchers['position'] = ['SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP']
    batters = data.get_batter_profile_data()

    # Act
    actual = service.get_pitcher_data_for_model(pitchers.copy(), batters)

    # Assert
    assert 'projected_matchup_strikeouts' in actual.columns
    assert 'opposing_woba' in actual.columns
    assert 'pitcher_era' in actual.columns

  def test_clean_and_normalize_and_find_best_match(self):
    # Arrange
    name = 'M. Busch'
    pitcher_profile_df = data.get_pitcher_profile_data()

    # Act
    cleaned = service.clean_and_normalize(name)
    best_match, score = service.find_best_match('D. May', pitcher_profile_df)

    # Assert
    assert cleaned == 'm busch'
    assert isinstance(best_match, str)
    assert isinstance(score, float) or isinstance(score, int)

  def test_simulate_slate_game_outcomes_adjusts_projected_signals(self):
    # Arrange
    hitters = data.get_batter_profile_data()
    hitters['game_id'] = hitters.apply(lambda r: f"{r['team']}_{r['opposing_team']}", axis=1)
    hitters['pitcher_target_index'] = 1.05
    hitters['team_win_probability'] = 0.58
    hitters['game_implied_total'] = 8.5
    hitters['park_factor'] = 100.0

    # Act
    out = service.simulate_slate_game_outcomes(hitters.copy(), simulations=50, random_state=1)

    # Assert
    assert 'simulated_game_strength_index' in out.columns
    assert 'simulated_game_outcome_multiplier' in out.columns
    assert out['simulated_game_strength_index'].between(0.82, 1.25).all()
    assert out['simulated_game_outcome_multiplier'].between(0.82, 1.30).all()

  def test_process_batter_data_for_model(self):
    # Arrange
    batters = data.get_batter_profile_data().copy()

    # Act
    actual = service.process_batter_data_model(batters)

    # Assert
    assert 'final_weighted_projection_signal' in actual.columns
    assert 'pitcher_weakness_ceiling_multiplier' in actual.columns

  @patch('util.data_util.get_league_batting_averages')
  def test_handle_pitcher_null_metrics(self, mock_league_batting_averages):
    # Arrange
    pitchers = data.get_pitcher_profile_data()
    mock_league_batting_averages.return_value = data.get_league_batting_averages()

    pitchers['team'] = ['STL', 'TBR', 'DET', 'CWS', 'MIL', 'CHC', 'KCR']
    pitchers['position'] = ['SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP']
    pitchers['pitcher_projected_outs'] = [7.1, 8.4, 9.3, 7.7, 10.3, 5.2, 6.3]
    pitchers['opposing_team_implied_total'] = [8.0, 11.0, 7.5, 9.0, 6.5, 7.5, 5.5]
    pitchers['opposing_woba'] = [None, None, None, None, None, None, None]
    pitchers['opposing_xwoba'] = [None, None, None, None, None, None, None]

    # Act
    actual = service.handle_pitcher_null_metrics(pitchers)

    # Assert
    assert actual['pitcher_projected_outs'].notnull().all()
    assert actual['pitcher_era'].notnull().all()
    assert actual['pitcher_k_9'].notnull().all()
    assert actual['opposing_team_implied_total'].notnull().all()
    assert actual['opposing_woba'].notnull().all()
    assert actual['opposing_xwoba'].notnull().all()