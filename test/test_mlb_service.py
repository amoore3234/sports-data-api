import unittest
from unittest.mock import patch
import pandas as pd

import service.mlb_service as service
import test_data as data

class TestMlbService(unittest.TestCase):

  @patch('util.data_util.get_starting_lineup')
  def test_get_starting_pitchers(self, mock_get_starting_lineup):

    #Arrange
    expected = data.get_pitcher_profile_test()
    mock_get_starting_lineup.return_value = data.get_starting_pitchers_test()

    #Act
    actual = service.get_starting_batters_and_pitchers(expected)

    #Assert
    assert len(actual) == 5
    assert expected['pitcher_name'].all() == actual['pitcher_name'].all()

  @patch('service.mlb_service.get_list_of_hitters')
  def test_get_starting_batters(self, mock_get_list_of_hitters):

    #Arrange
    expected = data.get_batter_profile_test()
    mock_get_list_of_hitters.return_value = list(data.get_starting_hitters_test()['Starting Lineup'])

    #Act
    actual = service.get_starting_batters_and_pitchers(expected)

    #Assert
    assert len(actual) == 9
    assert expected['batter_name'].all() == actual['batter_name'].all()

  @patch('service.mlb_service.get_list_of_hitters')
  def test_generate_top_order_starters(self, mock_get_list_of_hitters):
    #Arrange
    hitter_lineup_df = data.get_batter_profile_test()
    hitters_list = list(data.get_starting_hitters_test()['Starting Lineup'])
    mock_get_list_of_hitters.return_value = hitters_list
    expected = pd.DataFrame({
      'batter_id': [1, 2, 3, 6, 7, 8],
      'batter_name': ['J. Wetherholt', 'Ivan Herrera', 'A. Burleson', 'Colt Keith', 'K. McGonigle', 'G. Torres'],
      'batter_team': ['STL', 'STL', 'STL', 'DET', 'DET', 'DET']
    })

    #Act
    actual = service.generate_top_order_starters(hitter_lineup_df)

    #Assert
    assert len(actual) == 6
    assert expected['batter_name'].all() == actual['batter_name'].all()

  @patch('util.data_util.get_starting_lineup')
  def test_get_list_of_hitters_one_position(self, mock_get_starting_lineup):
    #Arrange
    position = 'C'
    mock_get_starting_lineup.return_value = data.get_starting_hitters_test()
    expected = ['C Pedro Pages R', 'C D. Dingler R']

    #Act
    actual = service.get_list_of_hitters(position)

    #Assert
    assert len(actual) == 2
    assert expected == actual

@patch('util.data_util.get_starting_lineup')
def test_get_list_of_hitters_multiple_position(mock_get_starting_lineup):
  #Arrange
  position = '1B|2B'
  mock_get_starting_lineup.return_value = data.get_starting_hitters_test()
  expected = ['2B J. Wetherholt L', '1B A. Burleson L', '2B G. Torres R', '1B S. Torkelson R']

  #Act
  actual = service.get_list_of_hitters(position)

  #Assert
  assert len(actual) == 4
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