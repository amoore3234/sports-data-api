import pandas as pd

def get_pitcher_profile_test():
  return pd.DataFrame({
    'pitcher_id': [1, 2, 3, 4, 5],
    'pitcher_name': ['Dustin May', 'Jack Flaherty', 'Lazaro Estrada', 'Anthony Kay', 'Chad Patrick'],
    'pitcher_team': ['STL', 'DET', 'TOR', 'CWS', 'MIL']
  })

def get_starting_pitchers_test():
  return pd.DataFrame({
    'Starting Lineup': ['D. May R', 'Jack Flaherty R', 'Lazaro Estrada R', 'Anthony Kay L', 'Chad Patrick R']
  })
