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
    'differential': data.get('to diff')
    }
    updated_data.append(modified_data)
    convert_data = json.dumps(updated_data, indent=2)

  return convert_data