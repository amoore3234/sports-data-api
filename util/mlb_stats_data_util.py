import datetime
import pandas as pd
from rapidfuzz import process, fuzz
import util.data_util as data
import requests

def get_mlb_batting_national_averages() -> dict:
  """Calculates the national averages for MLB hitting.

  Returns:
    dict: A dictionary containing the national averages for various MLB statistics.
  """
  batting_stat = data.get_league_batting_averages()

  batting_averages = {
    'league_batting_wOBA_average': (batting_stat['wOBA'] * batting_stat['PA']).sum() / batting_stat['PA'].sum(),
    'league_batter_BABIP_average': calculate_average_babip(batting_stat),
    'league_batter_ISO_average': calculate_average_iso(batting_stat)
  }

  return batting_averages

def calculate_average_babip(batting_stats):
  league_hits = batting_stats['H'].sum()
  league_home_runs = batting_stats['HR'].sum()
  league_at_bats = batting_stats['AB'].sum()
  league_strikeouts = batting_stats['SO'].sum()
  league_sacrafice_flies = batting_stats['SF'].sum()

  return (league_hits - league_home_runs) / (league_at_bats - league_strikeouts - league_home_runs + league_sacrafice_flies)

def calculate_average_iso(batting_stats):
  league_average_2b = batting_stats['2B'].sum()
  league_average_3b = batting_stats['3B'].sum()
  league_average_home_run = batting_stats['HR'].sum()
  league_average_at_bats = batting_stats['AB'].sum()

  return ((1 * league_average_2b) + (2 * league_average_3b) + (3 * league_average_home_run)) / league_average_at_bats

def get_mlb_pitcher_profile() -> list[dict]:
  """Creates statistics for a MLB pitcher.

  Returns:
    list: A list statistics for multiple MLB pitchers around the league.
  """
  # Prepare the datasets.
  statcast_data = data.get_statcast_data()
  pitcher_rank = data.get_statcast_pitching_stats()
  pitcher_stats_data = pitcher_rank.to_dict(orient='records')
  pitcher_statistics = data.get_pitching_stats()

  pitcher_profiles = []

  for pitcher_profile in pitcher_stats_data:
    pitcher_name = pitcher_profile.get('player_name')
    name_split = pitcher_name.split(',')
    name = f"{name_split[1]} {name_split[0]}"

    # Generate advance metrics for a pitcher's profile.
    profile = {
      'player_id': pitcher_profile.get('player_id', 658493 + 1),
      'name': name,
      'pitcher_hard_hit_percent': pitcher_profile.get('hard_hit_percent', 0) / 100,
      'pitcher_whiff_percent': pitcher_profile.get('whiff_percent', 0) / 100,
      'pitcher_fastball_velocity': pitcher_profile.get('fastball_avg_speed', 0),
      'pitcher_fastball_spin': pitcher_profile.get('fastball_avg_spin', 0),
      'pitcher_strike_K_percent': pitcher_profile.get('k_percent', 0) / 100,
      'pitcher_strike_K_BB_percent': pitcher_profile.get('bb_percent', 0) / 100,
      'pitcher_expected_xera': pitcher_profile.get('p_era', 0),
      'pitcher_woba_allowed': pitcher_profile.get('woba', 0),
      'pitcher_fly_ball_percent': pitcher_profile.get('flyballs_percent', 0) / 100
    }
    pitcher_profiles.append(profile)

  pitcher_data = statcast_data[['pitcher', 'p_throws', 'stand']].drop_duplicates()
  pitcher_advanced_data = pitcher_statistics[[
    'Name',
    'WHIP',
    'ERA',
    'Team',
    'BB/9',
    'Barrel%',
    'EV',
    'xFIP',
    'SIERA',
    'IP',
    'K/9',
    'H',
    'HBP',
    'GS'
  ]].drop_duplicates()

  # Include additional metrics to a pitcher's profile from different datasets.
  for pitcher in pitcher_profiles:
    pitcher_advanced_data['Name_Clean'] = pitcher_advanced_data['Name'].astype(str).str.strip().str.lower()

    # Clean the lookup variable
    lookup_name = str(pitcher['name']).strip().lower()
    found_pitcher_name = pitcher_advanced_data.loc[pitcher_advanced_data['Name_Clean'] == lookup_name]
    if not found_pitcher_name.empty:
      pitcher['pitcher_era'] = found_pitcher_name.iloc[0]['ERA']
      pitcher['pitcher_whip'] = found_pitcher_name.iloc[0]['WHIP']
      pitcher['pitcher_barrel_percent'] = found_pitcher_name.iloc[0]['Barrel%']
      pitcher['pitcher_bb_per_9'] = found_pitcher_name.iloc[0]['BB/9']
      pitcher['pitcher_xfip'] = found_pitcher_name.iloc[0]['xFIP']
      pitcher['pitcher_siera'] = found_pitcher_name.iloc[0]['SIERA']
      pitcher['pitcher_avg_exit_velocity'] = found_pitcher_name.iloc[0]['EV']
      pitcher['pitcher_innings_pitched'] = found_pitcher_name.iloc[0]['IP']
      pitcher['pitcher_k_9'] = found_pitcher_name.iloc[0]['K/9']
      pitcher['pitcher_hits_allowed'] = found_pitcher_name.iloc[0]['H']
      pitcher['pitcher_hit_hitters'] = found_pitcher_name.iloc[0]['HBP']
      pitcher['pitcher_game_starts'] = found_pitcher_name.iloc[0]['GS']
    else:
      # Fallback values
      pitcher['pitcher_era'] = pitcher_advanced_data['ERA'].mean()
      pitcher['pitcher_whip'] = pitcher_advanced_data['WHIP'].mean()
      pitcher['pitcher_barrel_percent'] = pitcher_advanced_data['Barrel%'].mean()
      pitcher['pitcher_bb_per_9'] = pitcher_advanced_data['BB/9'].mean()
      pitcher['pitcher_xfip'] = pitcher_advanced_data['xFIP'].mean()
      pitcher['pitcher_siera'] = pitcher_advanced_data['SIERA'].mean()
      pitcher['pitcher_avg_exit_velocity'] = pitcher_advanced_data['EV'].mean()
      pitcher['pitcher_innings_pitched'] = pitcher_advanced_data['IP'].mean().round(1)
      pitcher['pitcher_k_9'] = pitcher_advanced_data['K/9'].mean()
      pitcher['pitcher_hits_allowed'] = pitcher_advanced_data['H'].mean()
      pitcher['pitcher_hit_hitters'] = pitcher_advanced_data['HBP'].mean()
      pitcher['pitcher_game_starts'] = pitcher_advanced_data['GS'].mean()

    found_pitcher_id = pitcher_data.loc[pitcher_data['pitcher'] == pitcher['player_id'], 'p_throws']
    if not found_pitcher_id.empty:
      pitcher['pitcher_throwing_hand'] = found_pitcher_id.iloc[0]
    else:
      pitcher['pitcher_throwing_hand'] = 'R'

  pitcher_profile_df = pd.DataFrame(pitcher_profiles)

  return pitcher_profile_df

def get_mlb_batting_profile() -> dict:
  """Creates statistics for a MLB batter.

  Returns:
    list: A list statistics for multiple MLB battings around the league.
  """

  # Prepare the datasets.
  batting_rank = data.get_batting_stats()
  batting_stats_profile = batting_rank.to_dict(orient='records')
  statcast_data = data.get_statcast_data()

  batting_profiles = []

  # Generate advance metrics for a batting's profile.
  for batting_profile in batting_stats_profile:
    name_array = batting_profile.get('player_name').split()

    if len(name_array) >= 3:
      name = f"{name_array[2]} {name_array[0]} {name_array[1].rstrip(',')}"
    else:
      name = f"{name_array[1]} {name_array[0].rstrip(',')}"

    profile = {
      'batter_id': batting_profile.get('player_id'),
      'name': name,
      'batter_actual_wOBA': batting_profile.get('woba'),
      'batter_expected_xwOBA': batting_profile.get('xwoba'),
      'batter_BABIP': batting_profile.get('babip'),
      'batter_bat_speed': batting_profile.get('avg_swing_speed'),
      'batter_barrel_percent': batting_profile.get('barrel_batted_rate') / 100,
      'batter_ISO': batting_profile.get('isolated_power'),
      'singles': batting_profile.get('single'),
      'doubles': batting_profile.get('double'),
      'triples': batting_profile.get('triple'),
      'home_runs': batting_profile.get('home_run'),
      'rbis': batting_profile.get('b_rbi'),
      'runs_scored': batting_profile.get('r_run'),
      'walks': batting_profile.get('walk'),
      'hit_by_pitch': batting_profile.get('b_hit_by_pitch'),
      'stolen_bases': batting_profile.get('r_total_stolen_base'),
      'plate_appearances': batting_profile.get('pa')
    }

    batting_profiles.append(profile)

  batting_data = statcast_data[['player_name', 'batter', 'stand', 'events', 'bb_type', 'p_throws']].drop_duplicates()
  player_pool = list(batting_data['player_name'])

  # Include additional metrics to a batting's profile from different datasets.
  for batting in batting_profiles:
    name = batting['name']

    # Find the closest match
    best_match, score, idx = process.extractOne(name, player_pool, scorer=fuzz.WRatio)
    score = fuzz.token_sort_ratio(name, best_match)
    if score >= 75:
      batting_details = batting_data[batting_data['player_name'] == best_match]

      if not batting_details.empty:
        ids = batting_details['batter'].unique()

        if len(ids) > 0:
          batting['player_id'] = ids[0]

        # Generate a batting's platoon splits
        plate_appearance_data = batting_details.dropna(subset=['events']).copy()
        plate_appearance_data['is_hit'] = plate_appearance_data['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
        platoon_stats = plate_appearance_data.groupby('p_throws').agg(
          Plate_Appearance=('events', 'count'),
          Hits=('is_hit', 'sum')
        )
        platoon_stats_avg = platoon_stats['Hits'] / platoon_stats['Plate_Appearance']
        plate_appearance_stats_avg = platoon_stats_avg.to_dict()
        batting['platoon_vs_rhp'] = plate_appearance_stats_avg.get('R', 0)
        batting['platoon_vs_lhp'] = plate_appearance_stats_avg.get('L', 0)

        # Generate a batter's line drive rate
        line_drive = batting_details.dropna(subset=['bb_type'])
        line_drive_rate = (line_drive[line_drive['bb_type'] == 'line_drive']).shape[0] / batting_details.shape[0]
        batting['batting_line_drive_rate'] = line_drive_rate
        if batting['batting_line_drive_rate']:
          batting['batting_line_drive_rate'] = 0.20

      # Find a batter's batting stance
      found_match = batting_data.loc[batting_data['batter'] == batting['player_id'], 'stand']
      if not found_match.empty:
        batting['batting_stance'] = found_match.iloc[0]
      else:
        batting['batting_stance'] = 'R'

  batting_profile_df = pd.DataFrame(batting_profiles)

  batting_profile_df = batting_profile_df.dropna()

  return batting_profile_df

def fetch_mlb_live_stats(target_date):
  """
  Queries the official MLB Stats API for a 6-day trailing window up to the target_date,
  processes all box scores across all dates, and calculates total FanDuel DFS points.
  Format: 'YYYY-MM-DD'

  Args:
    target_date (str): The target date for which to fetch live stats.

  Returns:
    pd.DataFrame: A DataFrame containing the live stats for all players across the 6-day window.
  """
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
      'Accept': 'application/json'
  }

  # Calculate a clean 6-day window preceding your target date
  target_dt = datetime.datetime.strptime(target_date, '%Y-%m-%d').date()
  start_date = (target_dt - datetime.timedelta(days=6)).strftime('%Y-%m-%d')
  end_date = (target_dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

  # 1. Pull the multi-day schedule map in a single call
  schedule_url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&startDate={start_date}&endDate={end_date}"
  print(f"Fetching game schedule from {start_date} to {end_date}...")

  try:
    response = requests.get(schedule_url, headers=headers, timeout=10, verify=False)
    response.raise_for_status()
    schedule_data = response.json()
    dates_array = schedule_data.get('dates', [])
  except Exception as err:
    print(f"Network error occurred while fetching schedule: {err}")
    return pd.DataFrame()

  game_pks = []
  for date_block in dates_array:
    games = date_block.get('games', [])
    for game in games:
      if 'gamePk' in game:
        game_pks.append(game['gamePk'])

  print(f"Found {len(game_pks)} total games across the 6-day window. Parsing box scores...")
  all_player_stats = []

  # Iterate through each game's individual box score safely
  for game_pk in game_pks:
    boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
      box_resp = requests.get(boxscore_url, headers=headers, timeout=10, verify=False)
      box_data = box_resp.json()
    except Exception:
      continue  # Skip broken or postponed games safely

    # Parse both Home and Away teams
    for team_side in ['home', 'away']:
      team_data = box_data.get('teams', {}).get(team_side, {})
      team_name = team_data.get('team', {}).get('name', 'Unknown')
      players = team_data.get('players', {})

      for player_id, player_info in players.items():
        stats = player_info.get('stats', {})
        person = player_info.get('person', {})
        player_name = person.get('fullName')
        position = player_info.get('position', {}).get('abbreviation')

        if not stats:
          continue

        if 'batting' in stats and stats['batting'] != {}:
          b = stats['batting']
          if b.get('plateAppearances', 0) > 0:
            singles = b.get('hits', 0) - (b.get('doubles', 0) + b.get('triples', 0) + b.get('homeRuns', 0))

            all_player_stats.append({
              'player_id': player_id,
              'name': player_name,
              'team': team_name,
              'position': position,
              'player_type': 'Hitter',
              'current_singles': singles,
              'current_doubles': b.get('doubles', 0),
              'current_triples': b.get('triples', 0),
              'current_home_runs': b.get('homeRuns', 0),
              'current_base_on_balls': b.get('baseOnBalls', 0),
              'current_hit_by_pitch': b.get('hitByPitch', 0),
              'current_rbi': b.get('rbi', 0),
              'current_runs': b.get('runs', 0),
              'current_stolen_bases': b.get('stolenBases', 0),
              'current_plate_appearances': b.get('plateAppearances', 0)
            })

        if 'pitching' in stats and stats['pitching'] != {}:
          p = stats['pitching']
          if p.get('pitchesThrown', 0) > 0 or float(p.get('inningsPitched', '0.0')) > 0:

            all_player_stats.append({
              'player_id': player_id,
              'name': player_name,
              'team': team_name,
              'position': 'P',
              'player_type': 'Pitcher',
              'current_strikeouts': p.get('strikeOuts', 0),
              'current_earned_runs': p.get('earnedRuns', 0),
              'current_hits': p.get('hits', 0),
              'current_hits_by_pitch': p.get('hitsByPitch', 0),
              'current_base_on_balls': p.get('baseOnBalls', 0),
              'current_innings_pitched': float(p.get('inningsPitched', '0.0'))
            })

  if not all_player_stats:
    print("No player stats gathered.")
    return pd.DataFrame()

  return pd.DataFrame(all_player_stats)


def calculate_on_field_pitcher_stats(df):
  """
  Calculates single-game fantasy points for pitchers based on box score metrics.

  Args:
    df (pd.DataFrame): A DataFrame containing pitcher box score metrics.

  Returns:
    pd.DataFrame: A DataFrame containing the calculated fantasy points for each pitcher.
  """
  # Ensure all inputs are numeric floats/integers to prevent NaN failures
  stat_cols = ['pitcher_innings_pitched', 'pitcher_k_9', 'pitcher_hits_allowed', 'pitcher_hit_hitters', 'pitcher_era', 'pitcher_bb_per_9', 'pitcher_game_starts']
  for col in stat_cols:
    if col in df.columns:
      df[col] = df[col].fillna(0)
    else:
      # If a minor column is missing completely, initialize it with 0
      df[col] = 0

  df['innings_pitched_per_game'] = df['pitcher_innings_pitched'] / df['pitcher_game_starts']
  df['hits_allowed_per_innings_pitched'] = df['pitcher_hits_allowed'] / df['pitcher_innings_pitched']
  df['hit_batter_per_innings_pitched'] = df['pitcher_hit_hitters'] / df['pitcher_innings_pitched']

  return df

def calculate_on_field_hitter_stats(df):
  """
  Calculates single-game fantasy points for hitters based on box score metrics.

  Args:
    df (pd.DataFrame): A DataFrame containing hitter box score metrics.

  Returns:
    pd.DataFrame: A DataFrame containing the calculated fantasy points for each hitter.
  """
  # Ensure all inputs are numeric floats/integers to prevent NaN failures
  stat_cols = ['singles', 'doubles', 'triples', 'home_runs', 'rbis', 'runs_scored', 'walks', 'hit_by_pitch', 'stolen_bases']
  for col in stat_cols:
    if col in df.columns:
      df[col] = df[col].fillna(0)
    else:
      # If a minor column is missing completely, initialize it with 0
      df[col] = 0
  df['singles_per_game'] = df['singles'] / df['plate_appearances']
  df['doubles_per_game'] = df['doubles'] / df['plate_appearances']
  df['triples_per_game'] = df['triples'] / df['plate_appearances']
  df['home_runs_per_game'] = df['home_runs'] / df['plate_appearances']
  df['rbis_per_game'] = df['rbis'] / df['plate_appearances']
  df['runs_scored_per_game'] = df['runs_scored'] / df['plate_appearances']
  df['walks_per_game'] = df['walks'] / df['plate_appearances']
  df['stolen_bases_per_game'] = df['stolen_bases'] / df['plate_appearances']

  return df