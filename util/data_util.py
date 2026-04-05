import pandas as pd

def get_starting_lineup():
  starting_lineup_data = pd.read_csv('mlb_data/confirmed_starting_lineups.csv')
  return pd.DataFrame(starting_lineup_data)