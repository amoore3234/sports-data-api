import pandas as pd
import json

def get_nfl_team_offense_stats():
  """Generates NFL team offense statistics.

  Returns:
      json: Offensinve team data.
  """
  df = pd.read_csv('nfl_data/nfl_team_offense.csv')
  df.columns = df.columns.str.lower()
  team_offense_data = df.to_dict(orient='records')
  updated_data = []

  for data in team_offense_data:
    modified_data = {
    'rank': data.get('#'),
    'team': data.get('team'),
    'pts_per_game': data.get('pts/g'),
    'points': data.get('pts'),
    'plays': data.get('plays'),
    'yards': data.get('yds'),
    'yds_per_play': data.get('yds/play'),
    'first_downs': data.get('1st dwn'),
    'third_downs': {
        'made': data.get('made'),
        'attempts': data.get('att'),
        'percentage': data.get('pct')
    },
    'red_zone': {
        'made': data.get('made.1'),
        'attempts': data.get('att.1'),
        'percentage': data.get('pct.1')
    },
    'penalties': data.get('pen'),
    'penalty_yards': data.get('pen yds'),
    'turnover_differential': data.get('to diff')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_nfl_team_defense_stats():
  """Generates NFL team defense statistics.

  Returns:
      json: Defensive team data.
  """
  df = pd.read_csv('nfl_data/nfl_team_defense.csv')
  df.columns = df.columns.str.lower()
  team_defense_data = df.to_dict(orient='records')
  updated_data = []

  for data in team_defense_data:
    modified_data = {
      'rank': data.get('rank'),
      'team': data.get('team'),
      'total_points': data.get('pts'),
      'total_plays': data.get('plays'),
      'total_yards': data.get('yds'),
      'yards_per_play': data.get('yds/play'),
      'first_downs_allowed': data.get('1st dwn'),
      'third_downs_allowed': {
          'made': data.get('made'),
          'attempts': data.get('att'),
          'percentage': data.get('pct')
      },
      'red_zone_allowed': {
          'made': data.get('made.1'),
          'attempts': data.get('att.1'),
          'percentage': data.get('pct.1')
      },
      'penalties': data.get('pen'),
      'penalty_yards': data.get('pen yds'),
      'turnover_differential': data.get('to diff')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_overall_weighted_defensive_average():
  """Calculates the overall weighted defensive average.

  Returns:
      float: The weighted defensive average value.
  """
  df = pd.read_csv('nfl_data/nfl_team_defense.csv')
  df.columns = df.columns.str.lower()
  total_teams = len(df)
  total_yards_per_play_allowed = df['yds/play'].sum()
  weighted_average = total_yards_per_play_allowed / total_teams
  return weighted_average

def get_nfl_player_snap_count():
  """Generates a player snap count report.

  Returns:
      json: A player's snap count details for the season.
  """
  df = pd.read_csv('nfl_data/nfl_snap_count.csv')
  df.columns = df.columns.str.lower()
  player_snap_data = df.to_dict(orient='records')
  updated_data = []

  for data in player_snap_data:
    modified_data = {
      'player': data.get('player'),
      'position': data.get('pos'),
      'team': data.get('team'),
      'games_played': data.get('games'),
      'total_snaps': data.get('snaps'),
      'total_snaps_per_game': data.get('snaps/gm'),
      'snap_percentage': data.get('snap %'),
      'rush_percentage': data.get('rush %'),
      'target_percentage': data.get('tgt %'),
      'touch_percentage': data.get('touch %'),
      'util_percentage': data.get('util %'),
      'total_fantasy_points': data.get('fantasy pts'),
      'points_per_100_snaps': data.get('pts/100 snaps')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_nfl_odds():
  """Generates NFL team odds data.

  Returns:
      json: Team odds for a given a game.
  """
  df = pd.read_csv('nfl_data/nfl_odds.csv')
  df.columns = df.columns.str.lower()
  odds_data = df.to_dict(orient='records')
  updated_data = []

  for data in odds_data:

    modified_data = {
      'team_name': data.get('team'),
      'team': data.get('teamabbrev'),
      'spread': data.get('spread'),
      'over_under': data.get('over-under')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_nfl_teams():
  """Generates NFL team data.

  Returns:
      json: Team details for all NFL teams.
  """
  df = pd.read_csv('nfl_data/nfl_teams.csv')
  df.columns = df.columns.str.lower()
  teams_data = df.to_dict(orient='records')
  updated_data = []

  for data in teams_data:
    modified_data = {
      'team_name': data.get('team'),
      'team': data.get('abbrev'),
      'opponent': data.get('opponent')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data