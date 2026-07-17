import pandas as pd

def get_starting_lineup() -> pd.DataFrame:
  """
  Reads the confirmed starting lineups from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the confirmed starting lineups.
  """
  starting_lineup_data = pd.read_csv('mlb_data/confirmed_starting_lineups.csv', encoding='cp1252')
  return pd.DataFrame(starting_lineup_data)

def get_ball_park_factors() -> pd.DataFrame:
  """
  Reads the ball park factors from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the ball park factors.
  """
  park_factors_data = pd.read_csv("mlb_data/mlb_park_factors.csv")
  return pd.DataFrame(park_factors_data)

def get_pitcher_lineup() -> pd.DataFrame:
  """
  Reads the pitcher lineup from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the pitcher lineup.
    """
  pitcher_lineup_data = pd.read_csv('mlb_data/pitcher_lineup.csv')
  return pd.DataFrame(pitcher_lineup_data)

def get_batting_lineup() -> pd.DataFrame:
  """
  Reads the batting lineup from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the batting lineup.
  """
  batting_lineup_data = pd.read_csv('mlb_data/batting_lineup.csv')
  return pd.DataFrame(batting_lineup_data)

def get_pitching_stats() -> pd.DataFrame:
  """
  Reads the pitching stats from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the pitching stats.
  """
  pitching_stats = pd.read_csv('mlb_data/pitching_stats.csv', encoding='cp1252')
  return pd.DataFrame(pitching_stats)

def get_statcast_pitching_stats() -> pd.DataFrame:
  """
  Reads the statcast pitching stats from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the statcast pitching stats.
  """
  pitching_stats = pd.read_csv('mlb_data/statcast_pitcher_percentile_ranks.csv', encoding='cp1252')
  return pd.DataFrame(pitching_stats)

def get_league_pitching_averages() -> pd.DataFrame:
  """
  Reads the league pitching averages from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the league pitching averages.
  """
  league_pitching_stats = pd.read_csv('mlb_data/league_pitching_average.csv')
  return pd.DataFrame(league_pitching_stats)

def get_batting_stats() -> pd.DataFrame:
  """
  Reads the batting stats from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the batting stats.
  """
  batting_stats = pd.read_csv('mlb_data/batting_stats.csv', encoding='cp1252')
  return pd.DataFrame(batting_stats)

def get_league_batting_averages() -> pd.DataFrame:
  """
  Reads the league batting averages from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the league batting averages.
  """
  league_batting_stats = pd.read_csv('mlb_data/league_batting_average.csv')
  return pd.DataFrame(league_batting_stats)

def get_statcast_data() -> pd.DataFrame:
  """
  Reads the statcast data from a CSV file and returns it as a DataFrame.

  Returns:
    pd.DataFrame: A DataFrame containing the statcast data.
  """
  statcast_stats = pd.read_csv('mlb_data/statcast_stats.csv', encoding='cp1252')
  return pd.DataFrame(statcast_stats)