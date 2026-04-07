import pandas as pd

def get_pitcher_profile_test():
  return pd.DataFrame({
    'pitcher_id': [1, 2, 3, 4, 5],
    'pitcher_name': ['Dustin May', 'Jack Flaherty', 'Lazaro Estrada', 'Anthony Kay', 'Chad Patrick'],
    'pitcher_team': ['STL', 'DET', 'TOR', 'CWS', 'MIL']
  })

def get_batter_profile_test():
  return pd.DataFrame({
    'batter_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'batter_name': [
      'J. Wetherholt',
      'Ivan Herrera',
      'A. Burleson',
      'Nolan Gorman',
      'J. Walker',
      'Colt Keith',
      'K. McGonigle',
      'G. Torres',
      'Riley Greene'
    ],
    'batter_team': ['STL', 'STL', 'STL', 'STL', 'STL', 'DET', 'DET', 'DET', 'DET']
  })

def get_starting_pitchers_test():
  return pd.DataFrame({
    'Starting Lineup': [
      'D. May R',
      'Jack Flaherty R',
      'Lazaro Estrada R',
      'Anthony Kay L',
      'Chad Patrick R',
      'Tony Bennet L'
    ]
  })

def get_starting_hitters_test():
  return pd.DataFrame({
    'Starting Lineup': [
      '2B J. Wetherholt L',
      'DH Ivan Herrera R',
      '1B A. Burleson L',
      '3B Nolan Gorman L',
      'RF J. Walker R',
      'SS T. Saggese R',
      'LF N. Church L',
      'C Pedro Pages R',
      'CF Victor Scott L',
      '3B Colt Keith L',
      'SS K. McGonigle L',
      '2B G. Torres R',
      'DH K. Carpenter L',
      'LF Riley Greene L',
      'C D. Dingler R',
      'RF Z. McKinstry L',
      '1B S. Torkelson R',
      'CF P. Meadows L',
    ]
  })