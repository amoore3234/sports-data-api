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

def get_nfl_teams():
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

def get_nfl_fantasy_projections():
  nfl_dk_salary_df = get_nfl_dk_salary_data()
  nfl_odds_df = get_nfl_odds()
  nfl_teams_df = get_nfl_teams()
  nfl_snap_count_df = get_nfl_player_snap_count()
  nfl_team_offense_df = get_nfl_team_offense_stats()
  nfl_team_defense_df = get_nfl_team_defense_stats()

  generated_projections = []
  projected_game_snaps = {}

  for salary_data in json.loads(nfl_dk_salary_df):
    projection = {}
    projection['position'] = salary_data['position']
    projection['player_name'] = salary_data['player']
    projection['team'] = salary_data['team']
    projection['dk_salary'] = salary_data['salary']

    for team_data in json.loads(nfl_teams_df):
      if projection['team'] == team_data['team']:
        projection['opponent'] = team_data['opponent']
        break
    
    for odds_data in json.loads(nfl_odds_df):
      if projection['team'] == odds_data['team']:
        projection['spread'] = odds_data['spread']
        projection['over_and_under'] = odds_data['over_under']
        projection['projected_team_score'] = (odds_data['over_under'] / 2) - (odds_data['spread'] / 2)

    for snap_count_data in json.loads(nfl_snap_count_df):
      if projection['player_name'] == snap_count_data['player']:
        projection['snap_percentage'] = snap_count_data['snap_percentage']
        projection['fantasy_points_per_snap'] = snap_count_data['points_per_100_snaps'] / 100
        break

    for team_defense_data in json.loads(nfl_team_defense_df):
      opponent_team_name_lookup = get_nfl_team_name(projection['opponent'])
      opponent_team_name = get_mascot_name(opponent_team_name_lookup)

      if opponent_team_name == team_defense_data['team']:
        defensive_weighted_average = get_overall_weighted_defensive_average()
        projection['opponent_weight'] = team_defense_data['yards_per_play'] / defensive_weighted_average
        break

    for team_offense_data in json.loads(nfl_team_offense_df):
      team_name_lookup = get_nfl_team_name(projection['team'])
      team_name = get_mascot_name(team_name_lookup)

      if team_name == team_offense_data['team']:
        if projection['team'] not in projected_game_snaps:
          total_team_points = team_offense_data['points']
          team_points_per_game = team_offense_data['pts_per_game']
          team_total_snaps = team_offense_data['plays']

          team_snaps_per_game = get_snap_average_calculation(total_team_points, team_points_per_game, team_total_snaps)
          projection['team_snaps_per_game'] = team_snaps_per_game
          projected_game_snaps[projection['team']] = team_snaps_per_game
      
      opponent_team_name_lookup = get_nfl_team_name(projection['opponent'])
      opponent_team_name = get_mascot_name(opponent_team_name_lookup)

      if opponent_team_name == team_offense_data['team']:
        if projection['opponent'] not in projected_game_snaps:
          total_opponent_points = team_offense_data['points']
          opponent_points_per_game = team_offense_data['pts_per_game']
          opponent_total_snaps = team_offense_data['plays']

          opponent_snaps_per_game = get_snap_average_calculation(total_opponent_points, opponent_points_per_game, opponent_total_snaps)
          projected_game_snaps[projection['opponent']] = opponent_snaps_per_game
    
    projection['team_snaps_per_game'] = projected_game_snaps[projection['team']]
    projection['opponent_snaps_per_game'] = projected_game_snaps[projection['opponent']]

    average_snap_count = (projected_game_snaps[projection['team']] + projected_game_snaps[projection['opponent']]) / 2
    projection['projected_game_snaps'] = average_snap_count

    fantasy_points_per_snap = projection.get('fantasy_points_per_snap')
    if fantasy_points_per_snap is not None :
      projection['projected_fantasy_points'] = (fantasy_points_per_snap * projection['projected_game_snaps']) * projection['opponent_weight']
    
    fantasy_points = projection.get('projected_fantasy_points')
    if fantasy_points is not None and fantasy_points != 0.0:
      projection['projected_fantasy_value'] = fantasy_points / (projection['dk_salary'] / 1000)

    generated_projections.append(projection)
  convert_data = json.dumps(generated_projections, indent=2)
  return convert_data

def get_snap_average_calculation(total_team_points, team_points_per_game, team_total_snaps):
  average_games_played = total_team_points / team_points_per_game
  team_snaps_per_game = team_total_snaps / average_games_played
  return team_snaps_per_game

def get_nfl_team_name(nfl_team_abbreviation) -> str:
  teams = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB':  'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC':  'Kansas City Chiefs',
    'LV':  'Las Vegas Raiders',
    'LAC':  'Los Angeles Chargers',
    'LAR':  'Los Angeles Rams',
    'MIA':  'Miami Dolphins',
    'MIN':  'Minnesota Vikings',
    'NE':   'New England Patriots',
    'NO':   'New Orleans Saints',
    'NYG':  'New York Giants',
    'NYJ':  'New York Jets',
    'PHI':  'Philadelphia Eagles',
    'PIT':  'Pittsburgh Steelers',
    'SEA':  'Seattle Seahawks',
    'SF':   'San Francisco 49ers',
    'TB':   "Tampa Bay Buccaneers",
    "TEN":  "Tennessee Titans",
    "WAS":  "Washington Commanders"
  }
  return teams.get(nfl_team_abbreviation)

def get_mascot_name(team_name) -> str:
  for i in range(len(team_name) - 1, -1, -1):
    if (team_name[i] == ' '):
      mascot_name = team_name[i + 1:]
      return mascot_name