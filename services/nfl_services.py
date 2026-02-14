import pandas as pd
import json

def get_nfl_team_offense_stats():
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
  df = pd.read_csv('nfl_data/nfl_team_defense.csv')
  df.columns = df.columns.str.lower()
  total_teams = len(df)
  total_yards_per_play_allowed = df['yds/play'].sum()
  weighted_average = total_yards_per_play_allowed / total_teams
  return weighted_average

def get_nfl_player_snap_count():
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

def get_nfl_fantasy_points_leaders():
  df = pd.read_csv('nfl_data/nfl_fantasy_points_total.csv')
  df.columns = df.columns.str.lower()
  fantasy_points_data = df.to_dict(orient='records')
  updated_data = []

  for data in fantasy_points_data:
    modified_data = {
      'rank': data.get('rk'),
      'name': data.get('name'),
      'team': data.get('team'),
      'position': data.get('pos'),
      'games_played': data.get('gms'),
      'passing': {
        'yards': data.get('yds'),
        'touchdowns': data.get('td'),
        'interceptions': data.get('int')
      },
      'rushing': {
        'yards': data.get('yds.1'),
        'touchdowns': data.get('td.1')
      },
      'receiving': {
        'yards': data.get('yds.2'),
        'touchdowns': data.get('td.2'),
        'receptions': data.get('rec')
      },
      'fantasy_points_per_game': data.get('fpts/g'),
      'total_fantasy_points': data.get('fpts')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_nfl_dk_salary_data():
  df = pd.read_csv('nfl_data/nfl_dk_salaries.csv')
  df.columns = df.columns.str.lower()
  dk_salary_data = df.to_dict(orient='records')
  updated_data = []

  for data in dk_salary_data:
    modified_data = {
      'player': data.get('name'),
      'player_name_id': data.get('name + id'),
      'salary': data.get('salary'),
      'team': data.get('teamabbrev'),
      'position': data.get('position')
    }
    updated_data.append(modified_data)
  convert_data = json.dumps(updated_data, indent=2)
  return convert_data

def get_nfl_odds():
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