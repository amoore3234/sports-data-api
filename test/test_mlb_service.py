import unittest
from unittest.mock import patch

import service.mlb_service as service
import test_data as data

class TestMlbService(unittest.TestCase):

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

@patch('service.mlb_service.get_list_of_hitters')
def test_get_list_of_hitters_multiple_position(mock_get_starting_lineup):
  #Arrange
  position = '1B'
  mock_get_starting_lineup.return_value = data.get_starting_hitters_test()
  expected = ['1B A. Burleson L', '2B J. Wetherholt L', '1B S. Torkelson R', '2B G. Torres R']

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