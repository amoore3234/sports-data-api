import unittest
from unittest.mock import patch
import pandas as pd

import service.mlb_service as service
import test_data as data

class TestMlbService(unittest.TestCase):

  @patch('service.mlb_service.generate_top_order_starters')
  @patch('service.mlb_service.generate_stack_lineup')
  @patch('util.data_util.get_starting_lineup')
  def test_confirmed_starting_lineups_stacking_dk_true(
    self, mock_generate_top_order_starters, mock_generate_stack_lineup, mock_get_starting_lineup):

    # Arrange
    expected_hitters_df = data.get_batter_profile_data()
    mock_generate_top_order_starters.return_value = data.get_top_order_starters()
    mock_generate_stack_lineup.return_value = data.get_hitter_stack_lineup_data_dk()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()
    # STL vs DET
    expected_starting_hitters = [
      'J. Wetherholt',
      'Ivan Herrera',
      'A. Burleson',
      'Nolan Gorman',
      'J. Walker',
      'T. Saggese',
      'N. Church',
      'Pedro Pages',
      'Victor Scott',
      'Colt Keith',
      'K. McGonigle',
      'G. Torres',
      'K. Carpenter',
      'Riley Greene',
      'D. Dingler',
      'Z. McKinstry',
      'S. Torkelson',
      'P. Meadows',
    ]

    # Act
    actual = service.confirmed_starting_lineups(expected_hitters_df, True)

    # Assert
    assert len(actual['batter_name']) == len(expected_starting_hitters)
    assert actual['batter_name'].isin(expected_starting_hitters).all()

  @patch('service.mlb_service.generate_top_order_starters')
  @patch('service.mlb_service.generate_stack_lineup')
  @patch('util.data_util.get_starting_lineup')
  def test_confirmed_starting_lineups_stacking_top_order_dk_true(
    self, mock_generate_top_order_starters, mock_generate_stack_lineup, mock_get_starting_lineup):

    # Arrange
    expected_hitters_df = data.get_batter_profile_data()
    mock_generate_top_order_starters.return_value = data.get_top_order_starters()
    mock_generate_stack_lineup.return_value = data.get_hitter_stack_lineup_top_order_data_dk()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()
    # STL vs DET - Top 4 hitters on each team
    expected_starting_hitters = [
      'J. Wetherholt',
      'Ivan Herrera',
      'A. Burleson',
      'Nolan Gorman',
      'Colt Keith',
      'K. McGonigle',
      'G. Torres',
      'K. Carpenter'
    ]

    # Act
    actual = service.confirmed_starting_lineups(expected_hitters_df, True)

    # Assert
    assert len(actual['batter_name']) == len(expected_starting_hitters)
    assert actual['batter_name'].isin(expected_starting_hitters).all()

  @patch('service.mlb_service.get_list_of_team_game_matchups')
  def test_generate_stack_lineups(self, mock_get_list_of_team_game_matchups):

    # Arrange
    top_order_players_df = data.get_top_order_starters()
    starting_lineup_df = data.get_starting_players_data()
    mock_get_list_of_team_game_matchups.return_value = data.get_list_of_game_matchups()

    # STL vs DET
    expected_starting_hitters = [
      'J. Wetherholt',
      'Ivan Herrera',
      'A. Burleson',
      'Nolan Gorman',
      'Colt Keith',
      'K. McGonigle',
      'G. Torres',
      'K. Carpenter'
    ]

    # Act
    actual = service.generate_stack_lineup(top_order_players_df, starting_lineup_df)

    # Assert
    assert len(actual['batter_name']) == len(expected_starting_hitters)
    assert actual['batter_name'].isin(expected_starting_hitters).all()

  @patch('util.data_util.get_starting_lineup')
  def test_confirmed_starting_lineups_dk_pitchers(self, mock_get_starting_lineup):

    # Arrange
    expected_pitchers_df = data.get_pitcher_profile_data_dk()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()
    players = ['D. May (4892886)']

    # Act
    actual = service.confirmed_starting_lineups(expected_pitchers_df, False)

    # Assert
    assert len(actual['pitcher_name']) == 6
    assert actual['Name + ID'].isin(players).any()

  @patch('util.data_util.get_starting_lineup')
  def test_confirmed_starting_lineups_fd_pitchers(self, mock_get_starting_lineup):

    # Arrange
    expected_pitchers_df = data.get_pitcher_profile_data_fd()
    fd_salary_df = data.get_player_salary_data_fd()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()
    players = ['128430-119414:D. May']

    # Act
    actual = service.confirmed_starting_lineups(expected_pitchers_df, False)

    # Assert
    assert len(actual['pitcher_name']) == 6
    assert actual['name_id'].isin(players).any()

  @patch('service.mlb_service.get_starting_batters_or_pitchers')
  def test_get_missing_hitters(self, mock_get_starting_batters_or_pitchers):

    # Arrange
    exptected_hitters_df = data.get_batter_profile_data()
    positions_to_exclude = ['C']
    exptected_hitters_df = exptected_hitters_df[~exptected_hitters_df['position'].isin(positions_to_exclude)]
    mock_get_starting_batters_or_pitchers.return_value = data.get_batter_profile_data()

    # Act
    actual = service.get_missing_hitters(exptected_hitters_df)

    # Assert
    assert len(actual) == 36
    assert exptected_hitters_df['batter_name'].all() == actual['batter_name'].all()

  @patch('util.data_util.get_starting_lineup')
  def test_get_starting_pitchers_dk(self, mock_get_starting_lineup):

    #Arrange
    expected = data.get_pitcher_profile_data_dk()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()

    #Act
    actual = service.get_starting_batters_or_pitchers(expected)

    #Assert
    assert len(actual) == 6
    assert expected['pitcher_name'].all() == actual['pitcher_name'].all()

  @patch('util.data_util.get_starting_lineup')
  def test_get_starting_pitchers_fd(self, mock_get_starting_lineup):

    #Arrange
    expected = data.get_pitcher_profile_data_fd()
    mock_get_starting_lineup.return_value = data.get_starting_players_data()

    #Act
    actual = service.get_starting_batters_or_pitchers(expected)

    #Assert
    assert len(actual) == 6
    assert expected['pitcher_name'].all() == actual['pitcher_name'].all()

  @patch('service.mlb_service.get_list_of_hitters')
  def test_get_starting_batters(self, mock_get_list_of_hitters):

    #Arrange
    expected = data.get_batter_profile_data()
    mock_get_list_of_hitters.return_value = list(data.get_starting_players_data()['Starting Lineup'])

    #Act
    actual = service.get_starting_batters_or_pitchers(expected)

    #Assert
    # Size of data frame. Index starts at 0.
    assert len(actual) == 35
    assert expected['batter_name'].all() == actual['batter_name'].all()

  @patch('service.mlb_service.get_list_of_hitters')
  def test_generate_top_order_starters(self, mock_get_list_of_hitters):
    #Arrange
    hitter_lineup_df = data.get_batter_profile_data()
    hitters_list = list(data.get_starting_players_data()['Starting Lineup'])
    mock_get_list_of_hitters.return_value = hitters_list
    expected = data.get_top_order_starters()

    #Act
    actual = service.generate_top_order_starters(hitter_lineup_df)

    #Assert
    assert len(actual) == 16
    assert expected['batter_name'].all() == actual['batter_name'].all()

  @patch('util.data_util.get_starting_lineup')
  def test_get_list_of_hitters_one_position(self, mock_get_starting_lineup):
    #Arrange
    position = 'C'
    mock_get_starting_lineup.return_value = data.get_starting_players_data()
    expected = ['C Pedro Pages R', 'C D. Dingler R', 'C Edgar Quero S', 'C S. Perez R']

    #Act
    actual = service.get_list_of_hitters(position)

    #Assert
    assert len(actual) == 4
    assert expected == actual

@patch('util.data_util.get_starting_lineup')
def test_get_list_of_hitters_multiple_position(mock_get_starting_lineup):
  #Arrange
  position = '1B|2B'
  mock_get_starting_lineup.return_value = data.get_starting_players_data()
  expected = [
    '2B J. Wetherholt L',
    '1B A. Burleson L',
    '2B G. Torres R',
    '1B S. Torkelson R',
    '2B C. Meidroth R',
    '1B M. Murakami L',
    '1B V. Pasquantino L',
    '2B J. India R'
  ]

  #Act
  actual = service.get_list_of_hitters(position)

  #Assert
  assert len(actual) == 8
  assert expected == actual

# @patch('service.mlb_service.get_pitcher_starters')
# def test_get_pitchers(mock_get_pitchers):

#   # Arrange
#   pitcher_data = data.get_pitcher_profile_test()
#   starting_lineup_data = data.get_starting_pitchers_test()
#   mock_get_pitchers.return_value = pitcher_data

#   # Act
#   actual = service.get_pitcher_starters(pitcher_data, starting_lineup_data)

#   # Assert
#   assert len(actual) == 5