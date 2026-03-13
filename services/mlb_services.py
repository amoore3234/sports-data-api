from pybaseball import statcast_pitcher_percentile_ranks, pitching_stats, batting_stats, statcast, cache, playerid_lookup, schedule_and_record
cache.enable()
import pandas as pd
import numpy as np
import warnings
import contextlib
import os

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
  statcast_data = statcast(start_dt='2025-03-30', end_dt='2025-05-02')

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

def get_mlb_batter_profile() -> dict:
  """Creates statistics for a MLB batter.

  Returns:
    list: A list statistics for multiple MLB batters around the league.
  """

  # Prepare the datasets.
  batter_rank = batting_stats(2025)
  batter_stats = batter_rank.to_dict(orient='records')
  statcast_data = statcast(start_dt='2025-03-30', end_dt='2025-05-02')

  batter_profiles = []

  # Generate advance metrics for a batter's profile.
  for batter_profile in batter_stats:
    profile = {
      'batter_id': batter_profile.get('IDfg'),
      'batter_name': batter_profile.get('Name'),
      'batter_team': batter_profile.get('Team'),
      'batter_actual_wOBA': batter_profile.get('wOBA'),
      'batter_expected_xwOBA': batter_profile.get('xwOBA'),
      'batter_BABIP': batter_profile.get('BABIP'),
      'batter_bat_speed': batter_profile.get('Spd'),
      'batter_barrel_percent': batter_profile.get('Barrel%'),
      'batter_ISO': batter_profile.get('ISO')
    }

    batter_profiles.append(profile)

  batter_data = statcast_data[['batter','stand', 'events', 'bb_type']].drop_duplicates()

  # Include additional metrics to a batter's profile from different datasets.
  for batter in batter_profiles:
    name_array = batter['batter_name'].split()

    id = silent_lookup(name_array[1], name_array[0])
    batter_details = batter_data.loc[batter_data['batter'] == id]
    if not batter_details.empty:

      # Generate a batter's platoon splits
      plate_appearance_data = batter_details.dropna(subset=['events'])
      plate_appearance_data['is_hit'] = plate_appearance_data['events'].isin(['single', 'double', 'triple', 'home_run'])
      platoon_stats = plate_appearance_data.groupby('stand').agg(
        Plate_Appearance=('events', 'count'),
        Hits=('is_hit', 'sum')
      )
      platoon_stats_avg = platoon_stats['Hits'] / platoon_stats['Plate_Appearance']
      plate_appearance_stats_avg = platoon_stats_avg.to_dict()
      batter['platoon_stats'] = plate_appearance_stats_avg

      # Generate a batter's line drive rate
      line_drive = batter_details.dropna(subset=['bb_type'])
      line_drive_rate = (line_drive[line_drive['bb_type'] == 'line_drive']).shape[0] / batter_details.shape[0]
      batter['batter_line_drive_rate'] = line_drive_rate

    # Find a batter's batting stance
    found_match = batter_data.loc[batter_data['batter'] == id, 'stand']
    if not found_match.empty:
      batter['batter_stance'] = found_match.iloc[0]
  print(f"Batter Profile: {batter_profiles[0]}")
  return batter_profiles

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

def get_mlb_batter_national_averages() -> dict:
  """Calculates the national averages for MLB batters.

  Returns:
    dict: A dictionary containing the national averages for various MLB statistics.
  """
  batter_stats = batting_stats(2025)

  batting_averages = {
    'league_batting_wOBA_average': (batter_stats['wOBA'] * batter_stats['PA']).sum() / batter_stats['PA'].sum(),
    'league_batting_BABIP_average': calculate_average_babip(batter_stats),
    'league_batting_ISO_average': calculate_average_iso(batter_stats)
  }

  return batting_averages

def get_mlb_park_stats() -> list[dict]:
  mlb_teams = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW',
    'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR',
    'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM',
    'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA',
    'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
  ]

  park_stats = []

  for team in mlb_teams:
    ball_park_factors = get_ball_park_factors(team, 2025)

    park_stat = {
      'team': team,
      'ball_park_factor': ball_park_factors
    }
    park_stats.append(park_stat)

  return park_stats

def get_ball_park_factors(team, year):
  """Calculates the ball park factors for MLB stadiums.

  Args:
    team (str): The abbreviation of an MLB team
    year (int): The year for which to calculate the ball park factors.

  Returns:
    dict: A dictionary containing the ball park factors for MLB stadiums.
  """
  schedule_data = schedule_and_record(year, team)

  # Calculate the average runs scored at home games
  home_games = schedule_data[schedule_data['Home_Away'] == 'Home']
  home_runs_per_game = (home_games['R'].sum() + home_games['RA'].sum()) / len(home_games)

  # Calculate the average runs scored at road games
  road_games = schedule_data[schedule_data['Home_Away'] == '@']
  road_runs_per_game = (road_games['R'].sum() + road_games['RA'].sum()) / len(road_games)

  ball_park_factor = home_runs_per_game / road_runs_per_game

  return ball_park_factor

def generate_mlb_lineup():
  pitcher_profiles = get_mlb_pitcher_profile()
  batter_profiles = get_mlb_batter_profile()
  pitcher_national_average = get_mlb_pitcher_national_averages()
  batter_national_average = get_mlb_batter_national_averages()
  # ball_park_factors = get_mlb_park_stats()
  salary_data = pd.read_csv('mlb_data/mlb_salaries.csv')
  salary_data_df = pd.DataFrame(salary_data)

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
  game_matchups_df = pd.DataFrame(game_matchups)
  print(f"Team Schedule: {game_matchups_df}")

  pitcher_profile_df = pd.DataFrame(pitcher_profiles)
  pitcher_lineup_df = pitcher_profile_df.dropna()

  batter_profile_df = pd.DataFrame(batter_profiles)
  batter_lineup_df = batter_profile_df.dropna()

  pitcher_lineup_df['elite_strikeout_K'] = pitcher_lineup_df['pitcher_strike_K_percent'] > pitcher_national_average['league_pitcher_k_bb_average']
  elite_pitchers = pitcher_lineup_df[pitcher_lineup_df['elite_strikeout_K'] == True]

  batter_lineup_df['elite_wOBA'] = batter_lineup_df['batter_expected_xwOBA'] > batter_national_average['league_batting_wOBA_average']
  elite_batters = batter_lineup_df[batter_lineup_df['elite_wOBA'] == True]

  salary_lookup = salary_data_df.set_index('Name')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Name')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Name')['Name + ID'].to_dict()
  team_lookup = salary_data_df.set_index('Name')['TeamAbbrev'].to_dict()

  pitcher_lineup_df['salary'] = elite_pitchers['pitcher_name'].map(salary_lookup)
  pitcher_lineup_df['position'] = elite_pitchers['pitcher_name'].map(position_lookup)
  pitcher_lineup_df['name_id'] = elite_pitchers['pitcher_name'].map(name_id_lookup)
  pitcher_lineup_df['pitcher_teamabbrev'] = elite_pitchers['pitcher_name'].map(team_lookup)
  pitcher_lineup_df = pitcher_lineup_df.dropna()

  batter_lineup_df['salary'] = elite_batters['batter_name'].map(salary_lookup)
  batter_lineup_df['position'] = elite_batters['batter_name'].map(position_lookup)
  batter_lineup_df['name_id'] = elite_batters['batter_name'].map(name_id_lookup)
  batter_lineup_df['batter_teamabbrev'] = elite_batters['batter_name'].map(team_lookup)
  batter_lineup_df = batter_lineup_df.dropna()

  lineup_count = 0
  lineup_list = []
  positions = []
  player_salary_cap = 50000
  while lineup_count < 2:
    starting_lineup = {}
    pitcher_indices = list(pitcher_lineup_df.index)
    player_salary = 0

    pitcher_one_idx = np.random.choice(pitcher_indices)
    starting_lineup['pitcher_one'] = pitcher_lineup_df.loc[pitcher_one_idx]['name_id']
    pitcher_one_team = pitcher_lineup_df.loc[pitcher_one_idx]['pitcher_teamabbrev']
    pitcher_home_one_team = next((d for d in game_matchups if d.get('home_team') == pitcher_one_team), None)
    pitcher_away_one_team = next((d for d in game_matchups if d.get('away_team') == pitcher_one_team), None)

    if pitcher_home_one_team:
      batter_lineup_df = batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_home_one_team.get('away_team')].index)
      batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_home_one_team.get('away_team')].index, inplace=True)

    if pitcher_away_one_team:
      batter_lineup_df = batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_away_one_team.get('home_team')].index)
      batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_away_one_team.get('home_team')].index, inplace=True)

    player_salary += pitcher_lineup_df.loc[pitcher_one_idx]['salary']
    pitcher_indices.remove(pitcher_one_idx)

    pitcher_two_idx = np.random.choice(pitcher_indices)
    starting_lineup['pitcher_two'] = pitcher_lineup_df.loc[pitcher_two_idx]['name_id']
    pitcher_two_team = pitcher_lineup_df.loc[pitcher_two_idx]['pitcher_teamabbrev']
    pitcher_home_two_team = next((d for d in game_matchups if d.get('home_team') == pitcher_two_team), None)
    pitcher_away_two_team = next((d for d in game_matchups if d.get('away_team') == pitcher_two_team), None)

    if pitcher_home_two_team:
      batter_lineup_df = batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_home_two_team.get('away_team')].index)
      batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_home_two_team.get('away_team')].index, inplace=True)

    if pitcher_away_two_team:
      batter_lineup_df = batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_away_two_team.get('home_team')].index)
      batter_lineup_df.drop(batter_lineup_df[batter_lineup_df['batter_teamabbrev'] == pitcher_away_two_team.get('home_team')].index, inplace=True)

    player_salary += pitcher_lineup_df.loc[pitcher_two_idx]['salary']
    pitcher_indices.remove(pitcher_two_idx)

    batter_indices = list(batter_lineup_df.index)
    batter_idx = np.random.choice(batter_indices)
    position = batter_lineup_df.loc[batter_idx]['position']
    catcher_starters = batter_lineup_df[batter_lineup_df['position'] == 'C']
    print(catcher_starters)
    while position != 'C':
      if len(position) > 2:
        multiple_positions = position.split('/')
        if multiple_positions[-1] == 'C':
          position = multiple_positions[-1]
          starting_lineup['catcher'] = batter_lineup_df.loc[batter_idx]['name_id']
          print(f"List of multiple positions: {multiple_positions}")
      else:
        batter_idx = np.random.choice(batter_indices)
        position = batter_lineup_df.loc[batter_idx]['position']

        if position == 'C':
          starting_lineup['catcher'] = batter_lineup_df.loc[batter_idx]['name_id']
      
      print(f"Searching for matching position: {position}")
    
    positions.append(position)

    lineup_list.append(starting_lineup)
    lineup_count += 1

  print(f"Positions: {positions}")
  print(f"Lineup List: {lineup_list}")

def calculate_average_babip(batter_stats):
  league_hits = batter_stats['H'].sum()
  league_home_runs = batter_stats['HR'].sum()
  league_at_bats = batter_stats['AB'].sum()
  league_strikeouts = batter_stats['SO'].sum()
  league_sacrafice_flies = batter_stats['SF'].sum()

  return (league_hits - league_home_runs) / (league_at_bats - league_strikeouts - league_home_runs + league_sacrafice_flies)

def calculate_average_iso(batter_stats):
  league_average_2b = batter_stats['2B'].sum()
  league_average_3b = batter_stats['3B'].sum()
  league_average_home_run = batter_stats['HR'].sum()
  league_average_at_bats = batter_stats['AB'].sum()

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