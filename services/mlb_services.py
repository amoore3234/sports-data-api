
import pandas as pd
import numpy as np
import warnings
import model.position_type as roster
import util.mlb_stats_util as stats_util
from collections import Counter

# Disable the SettingWithCopy/ChainedAssignment warnings
pd.options.mode.chained_assignment = None

# Suppress all other Python/Library warnings (Deprecation, etc.)
warnings.filterwarnings('ignore')

def generate_mlb_lineup():
  is_pitcher_friendly_park = False
  is_hitter_friendly_park = False
  is_confirmed_starters = True
  is_fanduel_lineup = True

  salary_data = pd.read_csv('mlb_data/dk_mlb_salaries.csv')
  salary_data_df = pd.DataFrame(salary_data)

  if is_fanduel_lineup:
    salary_data = pd.read_csv('mlb_data/fd_mlb_salaries.csv')
    salary_data_df = pd.DataFrame(salary_data)

  pitcher_profile_df = stats_util.load_mlb_pitching_profiles()
  pitcher_lineup_df = pitcher_profile_df.dropna()

  batting_profile_df = stats_util.load_mlb_batting_profiles()
  batting_lineup_df = batting_profile_df.dropna()

  if is_pitcher_friendly_park:
    pitcher_lineup_df = stats_util.get_pitcher_friendly_ballpark(pitcher_lineup_df)
    pitcher_lineup_df = pitcher_lineup_df.dropna()
    print(f"Pitcher ballpark: {pitcher_lineup_df}")

  if is_hitter_friendly_park:
    batting_lineup_df = stats_util.get_hitter_friendly_ballpark(batting_lineup_df)
    pitcher_lineup_df = stats_util.drop_pitchers_at_hitter_friendly_ballpark(pitcher_lineup_df, batting_lineup_df)

  generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, is_hitter_friendly_park, is_confirmed_starters, is_fanduel_lineup)

def generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, is_hitter_friendly_park, is_confirmed_starters, is_fanduel_lineup):
  elite_pitchers = apply_elite_statistical_pitcher_filters(pitcher_lineup_df)
  elite_hitters = apply_elite_statistical_batting_filters(batting_lineup_df)

  if is_fanduel_lineup:
    pitcher_lineup_df = generate_pitcher_starting_lineup_fd(salary_data_df, pitcher_lineup_df, elite_pitchers)
    batting_lineup_df = generate_batting_starting_lineup_fd(salary_data_df, batting_lineup_df, elite_hitters)
  else:
    pitcher_lineup_df = generate_pitcher_starting_lineup(salary_data_df, pitcher_lineup_df, elite_pitchers)
    batting_lineup_df = generate_batting_starting_lineup(salary_data_df, batting_lineup_df, elite_hitters)

  if is_confirmed_starters:
    pitcher_lineup_df = generate_confirmed_starting_lineups(pitcher_lineup_df)
    batting_lineup_df = generate_confirmed_starting_lineups(batting_lineup_df)
  print(f"Batting_lineup_df: {batting_lineup_df}")

  if is_hitter_friendly_park:
    pitcher_lineup_df = drop_pitchers(pitcher_lineup_df, batting_lineup_df, salary_data_df)
    print(f"Dropped pitchers: {pitcher_lineup_df}")

  batting_lineup_df = get_missing_hitters(batting_lineup_df, salary_data_df, is_fanduel_lineup)
  generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df, is_hitter_friendly_park, is_fanduel_lineup)

def generate_confirmed_starting_lineups(lineup_df):
  starting_lineup_data = pd.read_csv('mlb_data/confirmed_starting_lineups.csv')
  starting_lineup_df = pd.DataFrame(starting_lineup_data)
  # pitcher_testing = {
  #   'pitcher_name':['Michael McGreevy', 'Cristian Javier', 'Reynaldo Lopez'],
  #   'position': ['SP', 'SP', 'SP']
  # }

  # hitter_testing = {
  #   'batter_name':['J. Wetherholt', 'Ivan Herrera', 'A. Burleson', 'Masyn Winn'],
  #   'position':['2B', 'C', '1B', 'SS']
  # }

  # lineup_df = pd.DataFrame(pitcher_testing)
  print(f"Pitchers: {lineup_df}")
  dk_pitcher = (lineup_df['position'].str.contains('SP')).any()
  fd_pitcher = (lineup_df['position'].str.contains('P')).any()

  if dk_pitcher or fd_pitcher:
    return get_pitcher_starters(lineup_df, starting_lineup_df)
  else:
    return get_batter_starters(lineup_df, starting_lineup_df)

def get_missing_hitters(batting_lineup_df, salary_df, is_fanduel_lineup):
  new_list = []
  positions = {
    'C': 0,
    '1B': 0,
    '2B': 0,
    '3B': 0,
    'SS': 0,
    'OF': 0
  }
  print(f"Batting lineup: {batting_lineup_df}")
  starting_lineup_positions = list(batting_lineup_df['position'])
  print(f"Starting positions: {starting_lineup_positions}")
  for starting_position in starting_lineup_positions:
    if len(starting_position) >= 3:
      multi_position = starting_position.split('/')
      new_list.append(multi_position[0])
      new_list.append(multi_position[-1])

  merged_list = starting_lineup_positions + new_list
  print(f"Final list: {merged_list}")

  frequency_map = Counter(positions) + Counter(merged_list)
  print(f"Frequency map: {frequency_map}")

  for position in positions:
    if position not in frequency_map:
      if is_fanduel_lineup:
        new_salary_df = map_missing_players(salary_df, position, 'Nickname', 'Team', 'Position', 'Salary', 'Player ID + Player Name', 'AvgPointsPerGame')
      else:
        new_salary_df = map_missing_players(salary_df, position, 'Name', 'TeamAbbrev', 'Position', 'Salary', 'Name + ID', 'AvgPointsPerGame')
      print(f"Salary data: {new_salary_df}")
      batting_lineup_df = pd.concat([batting_lineup_df, new_salary_df], ignore_index=True)

  batting_lineup_df = batting_lineup_df.fillna('')

  return batting_lineup_df

def map_missing_players(salary_df, position, name, team_abbrev, batting_position, salary, name_id, average_pts_per_game):
  new_salary_df = salary_df[salary_df['Position'].str.contains(position)]
  new_salary_df['batting_name'] = new_salary_df[name]
  new_salary_df['batting_team'] = new_salary_df[team_abbrev]
  new_salary_df['position'] = new_salary_df[batting_position]
  new_salary_df['salary'] = new_salary_df[salary]
  new_salary_df['name_id'] = new_salary_df[name_id]
  average_fts_pts = new_salary_df[average_pts_per_game].mean()
  new_salary_df = new_salary_df[new_salary_df[average_pts_per_game] > average_fts_pts]
  return new_salary_df

def get_pitcher_starters(lineup_df, starting_lineup_df):
  starting_pitcher_list = []

  starting_lineups = list(starting_lineup_df['Starting Lineup'])
  for name in starting_lineups:
    name_array = name.split()
    if len(name_array) == 3:
      print(f"Split array: {name_array}")
      player_lastname = name_array[1]
      starting_pitcher_list.append(player_lastname)

  pitcher_lastname_lookup = '|'.join(starting_pitcher_list)

  lineup_df = lineup_df[lineup_df['pitcher_name'].str.contains(pitcher_lastname_lookup)]
  lineup_df.dropna()
  print(f"Pitcher starters: {lineup_df}")
  return lineup_df

def get_batter_starters(lineup_df, starting_lineup_df):
  positions = 'C|1B|2B|3B|SS|CF|LF|RF|DH'

  hitters_starting_lineup = starting_lineup_df[starting_lineup_df['Starting Lineup'].str.contains(positions)]
  hitters_list = list(hitters_starting_lineup)
  starting_hitter_list = []

  for batter in hitters_list:
    name_array = batter.split()
    if len(name_array) == 4:
      player_lastname = name_array[2]
      starting_hitter_list.append(player_lastname)

  hitter_lastname_lookup = '|'.join(starting_hitter_list)

  lineup_df = lineup_df[lineup_df['batting_name'].str.contains(hitter_lastname_lookup)]
  lineup_df.dropna()
  print(f"Batter starters: {lineup_df}")
  return lineup_df

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

def apply_elite_statistical_pitcher_filters(pitcher_lineup_df):
  pitcher_national_average = stats_util.get_mlb_pitcher_national_averages()

  pitcher_lineup_df['elite_strikeout_K'] = pitcher_lineup_df['pitcher_strike_K_percent'] > pitcher_national_average['league_pitcher_k_bb_average']
  elite_pitchers = pitcher_lineup_df[pitcher_lineup_df['elite_strikeout_K'] == True]

  return elite_pitchers

def apply_elite_statistical_batting_filters(batting_lineup_df):
  batting_national_average = stats_util.get_mlb_batting_national_averages()

  batting_lineup_df['elite_wOBA'] = batting_lineup_df['batting_expected_xwOBA'] > batting_national_average['league_batting_wOBA_average']
  elite_hitters = batting_lineup_df[batting_lineup_df['elite_wOBA'] == True]

  return elite_hitters

def generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df, is_pitcher_friendly_park, is_fanduel_lineup):
  lineup_count = 0
  lineup_list = []

  while lineup_count < 10:
    starting_lineup = {}
    pitcher_starting_lineup_df = pitcher_lineup_df.dropna()
    print(f"Pitcher lineup setup: {pitcher_starting_lineup_df}")
    batting_starting_lineup_df = batting_lineup_df.dropna()
    print(f"Hitter lineup setup: {batting_starting_lineup_df}")
    pitcher_indices = list(pitcher_starting_lineup_df.index)
    chosen_players = set()
    player_salary = []
    print(f"Debugging salary: {player_salary}")

    if len(pitcher_indices) > 0:
      pitcher_one_idx = np.random.choice(pitcher_indices)
      starting_lineup['pitcher_one'] = pitcher_starting_lineup_df.loc[pitcher_one_idx]['name_id']

      drop_hitters_against_pitchers(salary_data_df, pitcher_one_idx, pitcher_starting_lineup_df, batting_starting_lineup_df, player_salary, pitcher_indices, is_pitcher_friendly_park)

    if is_fanduel_lineup:
      multi_position = 'C|1B'
      starting_lineup['catcher/1B'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, multi_position)
    else:
      if len(pitcher_indices) > 0:
        pitcher_two_idx = np.random.choice(pitcher_indices)
        starting_lineup['pitcher_two'] = pitcher_starting_lineup_df.loc[pitcher_two_idx]['name_id']

        drop_hitters_against_pitchers(salary_data_df, pitcher_two_idx, pitcher_starting_lineup_df, batting_starting_lineup_df, player_salary, pitcher_indices, is_pitcher_friendly_park)
      starting_lineup['catcher'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.CATCHER.value)
      starting_lineup['first_base'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.FIRSTBASE.value)
    starting_lineup['second_base'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.SECONDBASE.value)
    starting_lineup['third_base'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.THIRDBASE.value)
    starting_lineup['short_stop'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.SHORTSTOP.value)
    starting_lineup['outfielder_one'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)
    starting_lineup['outfielder_two'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)
    starting_lineup['outfielder_three'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)

    if is_fanduel_lineup:
      # if len(pitcher_indices) > 0:
      #   pitcher_two_idx = np.random.choice(pitcher_indices)
      #   starting_lineup['util'] = pitcher_starting_lineup_df.loc[pitcher_two_idx]['name_id']
      starting_lineup['util'] = get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, 'C|1B|2B|3B|SS|OF|OF|OF')


    keys_exist = [k for k, v in starting_lineup.items() if v is None]
    salary_cap = sum(player_salary).astype(int)

    if is_fanduel_lineup:
      if keys_exist or salary_cap <= 35000 and salary_cap >= 34500:
        lineup_list.append(starting_lineup)
        lineup_count += 1
        starting_lineup['salary_cap'] = salary_cap
      else:
        if keys_exist or (salary_cap <= 50000 and salary_cap >= 49500):
          lineup_list.append(starting_lineup)
          lineup_count += 1
          starting_lineup['salary_cap'] = salary_cap
  print(f"Print lineups: {lineup_list}")
  final_lineup_df = pd.DataFrame(lineup_list)

  if is_fanduel_lineup:
    final_lineup_df.to_csv('mlb_data/fd_mlb_lineups.csv', index=False)
  else:
    final_lineup_df.to_csv('mlb_data/dk_mlb_lineups.csv', index=False)

  return lineup_list

def get_starting_player(player_salary, chosen_players, batting_starting_lineup_df, position_type):
  starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains(position_type)]
  indices = list(starters.index)
  if len(indices) > 0:
    idx = np.random.choice(indices)
    player = starters.loc[idx]['name_id']
    while player in chosen_players:
      idx = np.random.choice(indices)
      player = starters.loc[idx]['name_id']
    chosen_players.add(player)
    player_salary.append(batting_starting_lineup_df.loc[idx]['salary'])
    print(f"Player salary: {player_salary}")
    return player

def generate_pitcher_starting_lineup_fd(salary_data_df, pitcher_lineup_df, elite_pitchers):
  salary_lookup = salary_data_df.set_index('Nickname')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Nickname')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Nickname')['Player ID + Player Name'].to_dict()
  team_lookup = salary_data_df.set_index('Nickname')['Team'].to_dict()

  pitcher_lineup_df['salary'] = elite_pitchers['pitcher_name'].map(salary_lookup)
  pitcher_lineup_df['position'] = elite_pitchers['pitcher_name'].map(position_lookup)
  pitcher_lineup_df['name_id'] = elite_pitchers['pitcher_name'].map(name_id_lookup)
  pitcher_lineup_df['pitcher_teamabbrev'] = elite_pitchers['pitcher_name'].map(team_lookup)
  pitcher_lineup_df = pitcher_lineup_df[pitcher_lineup_df['position'].str.contains('P')]

  return pitcher_lineup_df

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

def generate_batting_starting_lineup_fd(salary_data_df, batting_lineup_df, elite_hitters):
  salary_lookup = salary_data_df.set_index('Nickname')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Nickname')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Nickname')['Player ID + Player Name'].to_dict()
  team_lookup = salary_data_df.set_index('Nickname')['Team'].to_dict()

  batting_lineup_df['salary'] = elite_hitters['batting_name'].map(salary_lookup)
  batting_lineup_df['position'] = elite_hitters['batting_name'].map(position_lookup)
  batting_lineup_df['name_id'] = elite_hitters['batting_name'].map(name_id_lookup)
  batting_lineup_df['batting_teamabbrev'] = elite_hitters['batting_name'].map(team_lookup)
  batting_lineup_df = batting_lineup_df.dropna()

  return batting_lineup_df

def generate_batting_starting_lineup(salary_data_df, batting_lineup_df, elite_hitters):
  salary_lookup = salary_data_df.set_index('Name')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Name')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Name')['Name + ID'].to_dict()
  team_lookup = salary_data_df.set_index('Name')['TeamAbbrev'].to_dict()

  batting_lineup_df['salary'] = elite_hitters['batting_name'].map(salary_lookup)
  batting_lineup_df['position'] = elite_hitters['batting_name'].map(position_lookup)
  batting_lineup_df['name_id'] = elite_hitters['batting_name'].map(name_id_lookup)
  batting_lineup_df['batting_teamabbrev'] = elite_hitters['batting_name'].map(team_lookup)
  batting_lineup_df = batting_lineup_df.dropna()

  return batting_lineup_df

def drop_hitters_against_pitchers(salary_data_df, pitcher_idx, pitcher_lineup_df, batting_lineup_df, player_salary, pitcher_indices, is_pitcher_friendly_park):
  game_matchups = get_list_of_team_game_matchups(salary_data_df)

  pitcher_matchup = pitcher_lineup_df.loc[pitcher_idx]['pitcher_teamabbrev']
  pitcher_home_team = next((home for home in game_matchups if home.get('home_team') == pitcher_matchup), None)
  pitcher_away_team = next((away for away in game_matchups if away.get('away_team') == pitcher_matchup), None)

  if is_pitcher_friendly_park:
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

  player_salary.append(pitcher_lineup_df.loc[pitcher_idx]['salary'])
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