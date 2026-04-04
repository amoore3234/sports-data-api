from unittest.mock import patch

import service.mlb_services as service
import test_data as data

@patch('service.mlb_services.get_pitcher.starters')
def test_get_pitchers(mock_get_pitchers):

  # Arrange
  pitcher_data = data.get_pitcher_profile_test()
  starting_lineup_data = data.get_starting_pitchers_test()
  mock_get_pitchers.return_value = pitcher_data

  # Act
  actual = service.get_pitcher_starters(pitcher_data, starting_lineup_data)

  # Assert
  assert len(actual) == 5