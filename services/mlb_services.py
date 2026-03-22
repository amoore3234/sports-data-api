from pybaseball import statcast_pitcher_percentile_ranks, pitching_stats, batting_stats, statcast, cache, playerid_lookup, schedule_and_record
from datetime import date
cache.enable()
import pandas as pd
import numpy as np
import warnings
import contextlib
import os
import model.position_type as roster

# Disable the SettingWithCopy/ChainedAssignment warnings
pd.options.mode.chained_assignment = None

# Suppress all other Python/Library warnings (Deprecation, etc.)
warnings.filterwarnings('ignore')

def get_mlb_pitcher_profile() -> list[dict]:
  """Creates statistics for a MLB pitcher.

  Returns:
    list: A list statistics for multiple MLB pitchers around the league.
  """

  # Prepare the datasets.
  pitcher_rank = statcast_pitcher_percentile_ranks(2025)
  pitcher_stats_data = pitcher_rank.to_dict(orient='records')
  pitcher_statistics = pitching_stats(2025)
  # The statcast function can be slow, so limit the date range to the last 10 days.
  # Modify once the season starts
  # end = date.today().isoformat()
  # start = (date.today() - timedelta(days=20)).isoformat()
  statcast_data = statcast(start_dt='2025-03-30', end_dt='2025-05-02', parallel=False)

  pitcher_profiles = []

  for pitcher_profile in pitcher_stats_data:
    # Modify the player names to be in the format of "First Last" instead of "Last, First"
    player_name = pitcher_profile.get('player_name')
    name_array = player_name.split()
    if len(name_array) > 0:
      if len(name_array) > 2:
        player_name = f"{name_array[1]} {name_array[0].replace(',', '')} {name_array[2]}"
      else:
        player_name = f"{name_array[1]} {name_array[0].replace(',', '')}"

    # Generate advance metrics for a pitcher's profile.
    profile = {
      'pitcher_id': pitcher_profile.get('player_id'),
      'pitcher_name': player_name,
      'pitcher_hard_hit_percent': pitcher_profile.get('hard_hit_percent') / 100,
      'pitcher_whiff_percent': pitcher_profile.get('whiff_percent') / 100,
      'pitcher_fastball_velocity': pitcher_profile.get('fb_velocity'),
      'pitcher_fastball_spin': pitcher_profile.get('fb_spin'),
      'pitcher_exit_velocity': pitcher_profile.get('exit_velocity'),
      'pitcher_strike_K_percent': pitcher_profile.get('k_percent') / 100,
      'pitcher_strike_K_BB_percent': pitcher_profile.get('bb_percent') / 100,
      'pitcher_expected_xERA': pitcher_profile.get('xera')
    }

    pitcher_profiles.append(profile)

  pitcher_data = statcast_data[['pitcher', 'p_throws', 'stand']].drop_duplicates()
  pitcher_advanced_data = pitcher_statistics[['Name', 'WHIP', 'ERA', 'Team', 'BB', 'IP']].drop_duplicates()

  # Include additional metrics to a pitcher's profile from different datasets.
  for pitcher in pitcher_profiles:
    found_pitcher_name = pitcher_advanced_data.loc[pitcher_advanced_data['Name'] == pitcher['pitcher_name']]
    if not found_pitcher_name.empty:
      pitcher['pitcher_team'] = found_pitcher_name.iloc[0]['Team']
      pitcher['pitcher_ERA'] = found_pitcher_name.iloc[0]['ERA']
      pitcher['pitcher_WHIP'] = found_pitcher_name.iloc[0]['WHIP']
      pitcher['pitcher_BB_per_9'] = (found_pitcher_name.iloc[0]['BB'] / found_pitcher_name.iloc[0]['IP']) * 9

    found_pitcher_id = pitcher_data.loc[pitcher_data['pitcher'] == pitcher['pitcher_id'], 'p_throws']
    if not found_pitcher_id.empty:
      pitcher['pitcher_throwing_hand'] = found_pitcher_id.iloc[0]

  return pitcher_profiles

def get_mlb_batting_profile() -> dict:
  """Creates statistics for a MLB batting.

  Returns:
    list: A list statistics for multiple MLB battings around the league.
  """

  # Prepare the datasets.
  batting_rank = batting_stats(2025)
  batting_stats_profile = batting_rank.to_dict(orient='records')
  statcast_data = statcast(start_dt='2025-03-30', end_dt='2025-05-02')

  batting_profiles = []

  # Generate advance metrics for a batting's profile.
  for batting_profile in batting_stats_profile:
    profile = {
      'batting_id': batting_profile.get('IDfg'),
      'batting_name': batting_profile.get('Name'),
      'batting_team': batting_profile.get('Team'),
      'batting_actual_wOBA': batting_profile.get('wOBA'),
      'batting_expected_xwOBA': batting_profile.get('xwOBA'),
      'batting_BABIP': batting_profile.get('BABIP'),
      'batting_bat_speed': batting_profile.get('Spd'),
      'batting_barrel_percent': batting_profile.get('Barrel%'),
      'batting_ISO': batting_profile.get('ISO')
    }

    batting_profiles.append(profile)

  batting_data = statcast_data[['batter','stand', 'events', 'bb_type']].drop_duplicates()

  # Include additional metrics to a batting's profile from different datasets.
  for batting in batting_profiles:
    name_array = batting['batting_name'].split()

    id = silent_lookup(name_array[1], name_array[0])
    batting_details = batting_data.loc[batting_data['batter'] == id]
    if not batting_details.empty:

      # Generate a batting's platoon splits
      plate_appearance_data = batting_details.dropna(subset=['events'])
      plate_appearance_data['is_hit'] = plate_appearance_data['events'].isin(['single', 'double', 'triple', 'home_run'])
      platoon_stats = plate_appearance_data.groupby('stand').agg(
        Plate_Appearance=('events', 'count'),
        Hits=('is_hit', 'sum')
      )
      platoon_stats_avg = platoon_stats['Hits'] / platoon_stats['Plate_Appearance']
      plate_appearance_stats_avg = platoon_stats_avg.to_dict()
      batting['platoon_stats'] = plate_appearance_stats_avg

      # Generate a batting's line drive rate
      line_drive = batting_details.dropna(subset=['bb_type'])
      line_drive_rate = (line_drive[line_drive['bb_type'] == 'line_drive']).shape[0] / batting_details.shape[0]
      batting['batting_line_drive_rate'] = line_drive_rate

    # Find a batting's batting stance
    found_match = batting_data.loc[batting_data['batter'] == id, 'stand']
    if not found_match.empty:
      batting['batting_stance'] = found_match.iloc[0]
  return batting_profiles

def get_mlb_pitcher_national_averages() -> dict:
  """Calculates the national averages for MLB pitchers.

  Returns:
    dict: A dictionary containing the national averages for various MLB statistics.
  """
  pitcher_stats = pitching_stats(2025)

  pitching_averages = {
    'league_pitcher_strike_K_average_percent': pitcher_stats['Strikes'].sum() / pitcher_stats['Pitches'].sum(),
    'league_pitcher_BB9_average': (pitcher_stats['BB'].sum() / pitcher_stats['IP'].sum()) * 9,
    'league_pitcher_WHIP_average': (pitcher_stats['H'].sum() + pitcher_stats['BB'].sum()) / pitcher_stats['IP'].sum(),
    'league_pitcher_ERA_average': (pitcher_stats['ER'].sum() / pitcher_stats['IP'].sum()) * 9,
    'league_pitcher_k_bb_average': (pitcher_stats['K-BB%'].sum()) / len(pitcher_stats)
  }

  return pitching_averages

def get_mlb_batting_national_averages() -> dict:
  """Calculates the national averages for MLB hitting.

  Returns:
    dict: A dictionary containing the national averages for various MLB statistics.
  """
  batting_stat = batting_stats(2025)

  batting_averages = {
    'league_batting_wOBA_average': (batting_stat['wOBA'] * batting_stat['PA']).sum() / batting_stat['PA'].sum(),
    'league_batting_BABIP_average': calculate_average_babip(batting_stat),
    'league_batting_ISO_average': calculate_average_iso(batting_stat)
  }

  return batting_averages

def generate_mlb_lineup():
  pitcher_friendly_park = False
  hitter_friendly_park = True
  salary_data = pd.read_csv('mlb_data/mlb_salaries.csv')
  salary_data_df = pd.DataFrame(salary_data)

  pitcher_profile_df = load_mlb_pitching_profiles()
  pitcher_lineup_df = pitcher_profile_df.dropna()

  batting_profile_df = load_mlb_batting_profiles()
  batting_lineup_df = batting_profile_df.dropna()

  if pitcher_friendly_park:
    pitcher_lineup_df = get_pitcher_friendly_ballpark(pitcher_lineup_df)
    pitcher_lineup_df = pitcher_lineup_df.dropna()
    print(f"Pitcher ballpark: {pitcher_lineup_df}")

  if hitter_friendly_park:
    batting_lineup_df = get_hitter_friendly_ballpark(batting_lineup_df)
    pitcher_lineup_df = drop_pitchers_at_hitter_friendly_ballpark(pitcher_lineup_df, batting_lineup_df)

  generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, pitcher_friendly_park, hitter_friendly_park)

def get_pitcher_friendly_ballpark(pitcher_df):
  ball_park_factors = pd.read_csv("mlb_data/mlb_park_factors.csv")
  ball_park_factors_df = pd.DataFrame(ball_park_factors)
  ball_park_pitcher = pd.merge(
    pitcher_df,
    ball_park_factors_df,
    left_on='pitcher_team',
    right_on='Team',
    how='left')
  ball_park_lookup = ball_park_factors.to_dict(orient='records')
  ball_park_average = ball_park_pitcher['Park Factor'].mean()
  pitcher_df = ball_park_pitcher.drop(ball_park_pitcher[ball_park_pitcher['Park Factor'] > ball_park_average].index)

  return pitcher_df

def get_hitter_friendly_ballpark(batting_df):
  ball_park_factors = pd.read_csv("mlb_data/mlb_park_factors.csv")
  ball_park_factors_df = pd.DataFrame(ball_park_factors)
  ball_park_pitcher = pd.merge(
    batting_df,
    ball_park_factors_df,
    left_on='batting_team',
    right_on='Team',
    how='left')
  ball_park_lookup = ball_park_factors.to_dict(orient='records')
  ball_park_average = ball_park_pitcher['Park Factor'].mean()
  batting_df = ball_park_pitcher.drop(ball_park_pitcher[ball_park_pitcher['Park Factor'] < ball_park_average].index)
  batting_df.dropna()
  print(f"Batting ballpark: {batting_df[['batting_name', 'batting_team']]}")
  return batting_df

def drop_pitchers_at_hitter_friendly_ballpark(pitcher_df, batting_df):
  teams_to_exclude = batting_df['batting_team'].unique()
  pitcher_df = pitcher_df[~pitcher_df['pitcher_team'].isin(teams_to_exclude)]
  pitcher_df.dropna()
  print(f"Drop pitcher lineup for ballpark: {pitcher_df}")
  return pitcher_df

def generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, pitcher_friendly_park, hitter_friendly_park):
  elite_pitchers = apply_elite_statistical_pitcher_filters(pitcher_lineup_df, batting_lineup_df)
  elite_hitters = apply_elite_statistical_batting_filters(pitcher_lineup_df, batting_lineup_df)

  pitcher_lineup_df = generate_pitcher_starting_lineup(salary_data_df, pitcher_lineup_df, elite_pitchers)
  print(f"Pitcher lineup: {pitcher_lineup_df}")
  batting_lineup_df = generate_batting_starting_lineup(salary_data_df, batting_lineup_df, elite_hitters)
  print(f"Batting_lineup_df: {batting_lineup_df}")

  if hitter_friendly_park == True:
    pitcher_lineup_df = drop_pitchers(pitcher_lineup_df, batting_lineup_df, salary_data_df)
    print(f"Dropped pitchers: {pitcher_lineup_df}")

  generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df, pitcher_friendly_park, hitter_friendly_park)

def drop_pitchers(pitcher_lineup_df, batting_lineup_df, salary_data_df):
  game_matchups = get_list_of_team_game_matchups(salary_data_df)

  teams = set(batting_lineup_df['batting_teamabbrev'])
  print(f"Batting teams: {teams}")
  seen = set()
  pitcher_matchups = list(pitcher_lineup_df['pitcher_teamabbrev'])
  print(f"Pitcher matchup: {pitcher_matchups}")
  for pitcher_matchup in pitcher_matchups:
    pitcher_away_team = next((away for away in game_matchups if away.get('away_team') == pitcher_matchup), None)
    print(f"Pitcher away: {pitcher_away_team}")

    if pitcher_away_team and pitcher_away_team.get('away_team') not in seen and pitcher_away_team.get('home_team') in teams:
      pitcher_lineup_df = pitcher_lineup_df.drop(pitcher_lineup_df[pitcher_lineup_df['pitcher_teamabbrev'] == pitcher_away_team.get('away_team')].index)
      print(f"Dropping pitcher...{pitcher_lineup_df}")
      seen.add(pitcher_away_team.get('away_team'))

  return pitcher_lineup_df


def apply_elite_statistical_pitcher_filters(pitcher_lineup_df, batting_lineup_df):
  pitcher_national_average = get_mlb_pitcher_national_averages()

  pitcher_lineup_df['elite_strikeout_K'] = pitcher_lineup_df['pitcher_strike_K_percent'] > pitcher_national_average['league_pitcher_k_bb_average']
  elite_pitchers = pitcher_lineup_df[pitcher_lineup_df['elite_strikeout_K'] == True]

  return elite_pitchers

def apply_elite_statistical_batting_filters(pitcher_lineup_df, batting_lineup_df):
  batting_national_average = get_mlb_batting_national_averages()

  batting_lineup_df['elite_wOBA'] = batting_lineup_df['batting_expected_xwOBA'] > batting_national_average['league_batting_wOBA_average']
  elite_hitters = batting_lineup_df[batting_lineup_df['elite_wOBA'] == True]

  return elite_hitters

def generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df, pitcher_friendly_park, hitter_friendly_park):
  lineup_count = 0
  lineup_list = []
  positions = []
  player_salary_cap = 50000
  while lineup_count < 4:
    starting_lineup = {}
    pitcher_starting_lineup_df = pitcher_lineup_df.dropna()
    print(f"Pitcher lineup setup: {pitcher_starting_lineup_df}")
    batting_starting_lineup_df = batting_lineup_df.dropna()
    pitcher_indices = list(pitcher_starting_lineup_df.index)
    player_salary = 0

    if len(pitcher_indices) > 0:
      pitcher_one_idx = np.random.choice(pitcher_indices)
      starting_lineup['pitcher_one'] = pitcher_starting_lineup_df.loc[pitcher_one_idx]['name_id']

      drop_hitters_against_pitchers(salary_data_df, pitcher_one_idx, pitcher_starting_lineup_df, batting_starting_lineup_df, player_salary, pitcher_indices, pitcher_friendly_park, hitter_friendly_park)

    if len(pitcher_indices) > 0:
      pitcher_two_idx = np.random.choice(pitcher_indices)
      starting_lineup['pitcher_two'] = pitcher_starting_lineup_df.loc[pitcher_two_idx]['name_id']

      drop_hitters_against_pitchers(salary_data_df, pitcher_two_idx, pitcher_starting_lineup_df, batting_starting_lineup_df, player_salary, pitcher_indices, pitcher_friendly_park, hitter_friendly_park)

      starting_lineup['catcher'] = catcher_starters = get_starting_player(batting_starting_lineup_df, roster.PositionType.CATCHER.value)

    first_base_starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains('1B')]
    first_base_indices = list(first_base_starters.index)
    if len(first_base_indices) > 0:
      first_base_idx = np.random.choice(first_base_indices)
      starting_lineup['first_base'] = first_base_starters.loc[first_base_idx]['name_id']

    second_base_starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains('2B')]
    second_base_indices = list(second_base_starters.index)
    if len(second_base_indices) > 0:
      second_base_idx = np.random.choice(second_base_indices)
      starting_lineup['second_base'] = second_base_starters.loc[second_base_idx]['name_id']

    third_base_starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains('3B')]
    third_base_indices = list(third_base_starters.index)
    if len(third_base_indices) > 0:
      third_base_idx = np.random.choice(third_base_indices)
      starting_lineup['third_base'] = third_base_starters.loc[third_base_idx]['name_id']

    short_stop_starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains('SS')]
    short_stop_indices = list(short_stop_starters.index)
    if len(short_stop_indices) > 0:
      short_stop_idx = np.random.choice(short_stop_indices)
      starting_lineup['short_stop'] = short_stop_starters.loc[short_stop_idx]['name_id']

    outfielder_starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains('OF')]
    outfielder_indices = list(outfielder_starters.index)
    if len(outfielder_indices) > 0:
      outfielder_one_idx = np.random.choice(outfielder_indices)
      starting_lineup['outfielder_one'] = outfielder_starters.loc[outfielder_one_idx]['name_id']
      outfielder_two_idx = np.random.choice(outfielder_indices)
      starting_lineup['outfielder_two'] = outfielder_starters.loc[outfielder_two_idx]['name_id']
      outfielder_three_idx = np.random.choice(outfielder_indices)
      starting_lineup['outfielder_three'] = outfielder_starters.loc[outfielder_three_idx]['name_id']

    lineup_list.append(starting_lineup)
    lineup_count += 1
  print(f"Print lineups: {lineup_list}")
  return lineup_list

def get_starting_player(batting_starting_lineup_df, position_type):
  starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains(position_type)]
  indices = list(starters.index)
  if len(indices) > 0:
    idx = np.random.choice(indices)
    return starters.loc[idx]['name_id']

def generate_pitcher_starting_lineup(salary_data_df, pitcher_lineup_df, elite_pitchers):
  salary_lookup = salary_data_df.set_index('Name')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Name')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Name')['Name + ID'].to_dict()
  team_lookup = salary_data_df.set_index('Name')['TeamAbbrev'].to_dict()

  pitcher_lineup_df['salary'] = elite_pitchers['pitcher_name'].map(salary_lookup)
  pitcher_lineup_df['position'] = elite_pitchers['pitcher_name'].map(position_lookup)
  pitcher_lineup_df['name_id'] = elite_pitchers['pitcher_name'].map(name_id_lookup)
  pitcher_lineup_df['pitcher_teamabbrev'] = elite_pitchers['pitcher_name'].map(team_lookup)
  pitcher_lineup_df = pitcher_lineup_df[pitcher_lineup_df['position'].str.contains('SP')]

  return pitcher_lineup_df

def generate_batting_starting_lineup(salary_data_df, batting_lineup_df, elite_hitters):
  salary_lookup = salary_data_df.set_index('Name')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Name')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Name')['Name + ID'].to_dict()
  team_lookup = salary_data_df.set_index('Name')['TeamAbbrev'].to_dict()

  batting_lineup_df['salary'] = elite_hitters['batting_name'].map(salary_lookup)
  batting_lineup_df['position'] = elite_hitters['batting_name'].map(position_lookup)
  batting_lineup_df['name_id'] = elite_hitters['batting_name'].map(name_id_lookup)
  batting_lineup_df['batting_teamabbrev'] = elite_hitters['batting_name'].map(team_lookup)

  return batting_lineup_df

def drop_hitters_against_pitchers(salary_data_df, pitcher_idx, pitcher_lineup_df, batting_lineup_df, player_salary, pitcher_indices, pitcher_friendly_park, hitter_friendly_park):
  game_matchups = get_list_of_team_game_matchups(salary_data_df)

  pitcher_matchup = pitcher_lineup_df.loc[pitcher_idx]['pitcher_teamabbrev']
  pitcher_home_team = next((home for home in game_matchups if home.get('home_team') == pitcher_matchup), None)
  pitcher_away_team = next((away for away in game_matchups if away.get('away_team') == pitcher_matchup), None)

  if pitcher_friendly_park == True:
    if pitcher_home_team:
      home_team = pitcher_home_team.get('home_team')
      get_team_matchup(home_team, batting_lineup_df)

    if pitcher_away_team:
      away_team = pitcher_away_team.get('away_team')
      get_team_matchup(away_team, batting_lineup_df)
  else:
    if pitcher_home_team:
      away_team = pitcher_home_team.get('away_team')
      get_team_matchup(away_team, batting_lineup_df)

    if pitcher_away_team:
      home_team = pitcher_away_team.get('home_team')
      get_team_matchup(home_team, batting_lineup_df)

  player_salary += pitcher_lineup_df.loc[pitcher_idx]['salary']
  pitcher_lineup_df.drop(pitcher_lineup_df[pitcher_lineup_df['pitcher_teamabbrev'] == pitcher_lineup_df.loc[pitcher_idx]['pitcher_teamabbrev']].index, inplace=True)

  pitcher_indices[:] = list(pitcher_lineup_df.index)

def get_list_of_team_game_matchups(salary_data_df) -> list[dict]:
  game_matchups = []

  game_schedule = salary_data_df['Game Info'].unique()
  for game in game_schedule:
    team_schedule = game.split(' ')
    team_matchup = team_schedule[0]
    teams = team_matchup.split('@')

    final_matchup = {
      'home_team': teams[1],
      'away_team': teams[0]
    }
    game_matchups.append(final_matchup)
    return game_matchups

def get_team_matchup(team_matchup, batting_lineup_df):
    batting_lineup_df.drop(batting_lineup_df[batting_lineup_df['batting_teamabbrev'] == team_matchup].index, inplace=True)

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

def silent_lookup(lastname, firstname):
  # Suppress the output for the playerid_lookup function
  with contextlib.redirect_stdout(open(os.devnull, 'w')):
    try:
      results = playerid_lookup(lastname, firstname, fuzzy=True)
      if not results.empty:
        results['mlb_played_last'] = pd.to_numeric(results['mlb_played_last'], errors='coerce')
        results = results.dropna(subset=['mlb_played_last'])
        id = results.sort_values('mlb_played_last', ascending=False).iloc[0]['key_mlbam']
        return id
    except Exception:
      return pd.DataFrame()

def load_mlb_batting_profiles():
  today = date.today()
  batting_profile_filename = f"mlb_data/batting_profile_{today}.csv"
  pitcher_profile_filename = f"pitcher_profile_{today}.csv"

  if os.path.exists(batting_profile_filename):
    print(f"Loading batting profiles...")
    batting_profile = pd.read_csv(batting_profile_filename)
    batting_profile_df = pd.DataFrame(batting_profile)
    return batting_profile_df
  else:
    print(f"Fetching and saving today's batting profiles...")
    add_batting_profile = get_mlb_batting_profile()
    add_batting_profile_df = pd.DataFrame(add_batting_profile)
    add_batting_profile_df.to_csv(batting_profile_filename, index=False)
    print(f"Successfully cached: {batting_profile_filename}")
    return add_batting_profile_df

def load_mlb_pitching_profiles():
  today = date.today()
  pitching_profile_filename = f"mlb_data/pitcher_profile_{today}.csv"

  if os.path.exists(pitching_profile_filename):
    print(f"Loading pitching profiles...")
    pitching_profile = pd.read_csv(pitching_profile_filename)
    pitching_profile_df = pd.DataFrame(pitching_profile)
    return pitching_profile_df
  else:
    print(f"Fetching and saving today's batting profiles...")
    add_pitching_profile = get_mlb_pitcher_profile()
    add_pitching_profile_df = pd.DataFrame(add_pitching_profile)
    add_pitching_profile_df.to_csv(pitching_profile_filename, index=False)
    print(f"Successfully cached: {pitching_profile_filename}")
    return add_pitching_profile_df