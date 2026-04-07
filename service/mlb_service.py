
import pandas as pd
import numpy as np
import warnings
import model.position_type as roster
import util.mlb_stats_util as stats_util
import util.data_util as data
from collections import Counter

# Disable the SettingWithCopy/ChainedAssignment warnings
pd.options.mode.chained_assignment = None

# Suppress all other Python/Library warnings (Deprecation, etc.)
warnings.filterwarnings('ignore')

def generate_mlb_lineup() -> dict:
  """Generates a MLB roster.

  Returns:
      json: A lineup that consists of 9 to 10 players.
  """
  is_pitcher_friendly_park = False
  is_hitter_friendly_park = False
  is_stacking = False
  is_confirmed_starters = True
  is_fanduel_lineup = True

  salary_data = pd.read_csv('mlb_data/dk_mlb_salaries.csv')
  salary_data_df = pd.DataFrame(salary_data)
  salary_data_df.rename(columns={'TeamAbbrev': 'Team'}, inplace=True)

  if is_fanduel_lineup:
    salary_data = pd.read_csv('mlb_data/fd_mlb_salaries.csv')
    salary_data_df = pd.DataFrame(salary_data)
    salary_data_df.rename(columns={'Nickname': 'Name', 'Player ID + Player Name': 'Name + ID', 'TeamAbbrev': 'Team', 'Game': 'Game Info'}, inplace=True)

  pitcher_profile_df = stats_util.load_mlb_pitching_profiles()
  pitcher_lineup_df = pitcher_profile_df.dropna()

  batting_profile_df = stats_util.load_mlb_batting_profiles()
  batting_lineup_df = batting_profile_df.dropna()

  # Return pitcher-friendly parks that favor pitchers.
  if is_pitcher_friendly_park:
    pitcher_lineup_df = stats_util.get_pitcher_friendly_ballpark(pitcher_lineup_df)
    pitcher_lineup_df = pitcher_lineup_df.dropna()

  # Return hitter-friendly parks that favor hitters.
  if is_hitter_friendly_park:
    batting_lineup_df = stats_util.get_hitter_friendly_ballpark(batting_lineup_df)
    pitcher_lineup_df = stats_util.drop_pitchers_at_hitter_friendly_ballpark(pitcher_lineup_df, batting_lineup_df)

  if is_stacking:
    weak_pitchers_df = generate_statistically_weak_pitchers(pitcher_lineup_df)
    weak_lineup_df = generate_starting_lineup(salary_data_df, pitcher_lineup_df, weak_pitchers_df, 'pitcher_name', 'pitcher_teamabbrev')
    print(f"Weaker pitchers: {weak_lineup_df}")

  generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, is_hitter_friendly_park,
                              is_confirmed_starters, is_fanduel_lineup)

def generate_elite_ball_players(pitcher_lineup_df, batting_lineup_df, salary_data_df, is_hitter_friendly_park,
                                is_confirmed_starters, is_fanduel_lineup):
  """Generates a list of top pitchers and hitters in the league.

  Parameters:
    pitcher_lineup_df: Data Frame containing a list of pitchers.
    batting_lineup_df: Data Frame containing a list of hitters.
    salary_data_df: Data Frame containing a list of players with their relative prices.
    is_hitter_friendly_park: Boolean for checking hitter ball park.
    is_confirmed_starters: Boolean to confirm starters.
    is_fanduel_lineup: Boolean to create fanduel lineup.

  Returns:
      void: Updates the pitcher and batting Data Frames.
  """

  elite_pitchers = apply_elite_statistical_pitcher_filters(pitcher_lineup_df)
  elite_hitters = apply_elite_statistical_batting_filters(batting_lineup_df)

  pitcher_lineup_df = generate_starting_lineup(salary_data_df, pitcher_lineup_df, elite_pitchers, 'pitcher_name', 'pitcher_teamabbrev')
  batting_lineup_df = generate_starting_lineup(salary_data_df, batting_lineup_df, elite_hitters, 'batter_name', 'batting_teamabbrev')

  if is_confirmed_starters:
    batting_lineup_df = generate_starting_lineup(salary_data_df, batting_lineup_df, batting_lineup_df, 'batter_name', 'batting_teamabbrev')
    print(f"Generate lineup: {batting_lineup_df}")
    pitcher_lineup_df = generate_confirmed_starting_lineups(pitcher_lineup_df, salary_data_df)
    batting_lineup_df = generate_confirmed_starting_lineups(batting_lineup_df, salary_data_df)

  if is_hitter_friendly_park:
    # Don't return pitchers at parks that favor hitters.
    pitcher_lineup_df = drop_pitchers(pitcher_lineup_df, batting_lineup_df, salary_data_df)

  batting_lineup_df = get_missing_hitters(batting_lineup_df, is_fanduel_lineup)
  generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df, is_hitter_friendly_park, is_fanduel_lineup)

def generate_confirmed_starting_lineups(lineup_df, salary_data_df) -> pd.DataFrame:
  """Confirms that players are starting in the game.

  Parameters:
    lineup_df: Pitcher or hitter Data Frame.

  Returns:
      DataFrame: Returns a new data frame with players that are officially starting in the game.
  """
  starting_lineup_data = pd.read_csv('mlb_data/confirmed_starting_lineups.csv')
  starting_lineup_df = pd.DataFrame(starting_lineup_data)
  is_stacking = False
  dk_pitcher = (lineup_df['position'].str.contains('SP')).any()
  fd_pitcher = (lineup_df['position'].str.contains('P')).any()

  if dk_pitcher or fd_pitcher:
    return get_pitcher_starters(lineup_df, starting_lineup_df)
  elif is_stacking:
    # hitter_profile_df = stats_util.load_mlb_batting_profiles()
    # batting_lineup_df = generate_starting_lineup(salary_data_df, hitter_profile_df, hitter_profile_df, 'batter_name', 'batting_teamabbrev')
    batting_starting_lineup = generate_top_order_starters(lineup_df)
    return generate_stack_lineup(batting_starting_lineup, starting_lineup_df, salary_data_df)
  else:
    return get_batter_starters(lineup_df)

def get_missing_hitters(batting_lineup_df, is_fanduel_lineup) -> pd.DataFrame:
  """Generates a list of hitters to fill in open positions.

  Parameters:
    batting_lineup_df: Data Frame containing a list of hitters.
    salary_df: Data Frame containing a list of players with their relative prices.
    is_fanduel_lineup: Boolean to create fanduel lineup.

  Returns:
      DataFrame: Returns a list of hitters to successfully create a lineup.
  """
  position_splits = []
  positions = {
    'C': 0,
    '1B': 0,
    '2B': 0,
    '3B': 0,
    'SS': 0,
    'OF': 0
  }

  starting_lineup_positions = list(batting_lineup_df['position'])
  # Consider hitters that play multiple positions. Isolate poisitions to return an accurate count.
  for starting_position in starting_lineup_positions:
    if len(starting_position) >= 3:
      multi_position = starting_position.split('/')
      position_splits.append(multi_position[0])
      position_splits.append(multi_position[-1])

  #Include the position splits with the current list of positions in the Data Frame.
  merged_list = starting_lineup_positions + position_splits

  frequency_map = Counter(positions) + Counter(merged_list)

  for position in positions:
    if position not in frequency_map:
      if is_fanduel_lineup:
        # new_hitter_roster_df = map_missing_players(
        #   salary_df, position, 'Nickname', 'Team', 'Position', 'Salary', 'Player ID + Player Name', 'AvgPointsPerGame')
        new_hitter_roster_df = return_missing_starters(batting_lineup_df, position)
      else:
        # new_hitter_roster_df = map_missing_players(
        #   salary_df, position, 'Name', 'TeamAbbrev', 'Position', 'Salary', 'Name + ID', 'AvgPointsPerGame')
        new_hitter_roster_df = return_missing_starters(batting_lineup_df, position)
      batting_lineup_df = pd.concat([batting_lineup_df, new_hitter_roster_df], ignore_index=True)

  batting_lineup_df = batting_lineup_df.fillna('')

  return batting_lineup_df

def map_missing_players(salary_df, position, name, team_abbrev, batting_position,
                        salary, name_id, average_pts_per_game) -> pd.DataFrame:
  """Map players to an existing Data Frame to return a new Data Frame of players.

  Parameters:
    salary_df: Data Frame containing a list of players with their relative prices.
    position: The vacant position.
    name: The column name for a player's name.
    team_abbrev: The column name for the team code.
    batting_position: The column name for player's listed position.
    salary: The column name for a player's salary.
    name_id: The column name for a player's id.
    average_pts_per_game: The column name for a player's average points per game.

  Returns:
      DataFrame: Returns a new data frame containing a new batting lineup.
  """
  new_batting_lineup_df = salary_df[salary_df['Position'].str.contains(position)]
  new_batting_lineup_df['batter_name'] = new_batting_lineup_df[name]
  new_batting_lineup_df['batter_team'] = new_batting_lineup_df[team_abbrev]
  new_batting_lineup_df['position'] = new_batting_lineup_df[batting_position]
  new_batting_lineup_df['salary'] = new_batting_lineup_df[salary]
  new_batting_lineup_df['name_id'] = new_batting_lineup_df[name_id]
  average_fts_pts = new_batting_lineup_df[average_pts_per_game].mean()
  new_batting_lineup_df = new_batting_lineup_df[new_batting_lineup_df[average_pts_per_game] > average_fts_pts]
  return new_batting_lineup_df

def get_starting_batters_and_pitchers(lineup_df):
  """Generate starting hitters and pitchers.

  Parameters:
    lineup_df: Pitcher or hitter Data Frame.

  Returns:
      DataFrame: Returns updated Data Frame.
  """

  if 'batter_name' in lineup_df.columns:
    positions = 'C|1B|2B|3B|SS|CF|LF|RF|DH'
    hitters_list = get_list_of_hitters(positions)

    return get_starters(lineup_df, hitters_list, column_name='batter_name', length_value=4, index=2)
  else:
    pitchers_list = list(data.get_starting_lineup()['Starting Lineup'])

    return get_starters(lineup_df, pitchers_list, column_name='pitcher_name', length_value=3, index=1)

def get_starters(lineup_df, starting_list, column_name, length_value, index):
  starting_lineup_list = []

  for starter in starting_list:
    name_array = starter.split()
    if len(name_array) == length_value:
      player_lastname = name_array[index]
      starting_lineup_list.append(player_lastname)

  lookup_table = '|'.join(starting_lineup_list)

  lineup_df = lineup_df[lineup_df[column_name].str.contains(lookup_table)]
  lineup_df = lineup_df.dropna()

  return lineup_df


def generate_top_order_starters(hitter_lineup_df):
  positions = 'C|1B|2B|3B|SS|CF|LF|RF|DH'

  hitters_list = get_list_of_hitters(positions)
  top_order_list = []
  start = 0
  end = 3

  while end <= len(hitters_list):
    starting_lineup = hitters_list[start:end]
    for batter in starting_lineup:
      name_array = batter.split()
      if len(name_array) == 4:
        player_lastname = name_array[2]
        top_order_list.append(player_lastname)
    start += 9
    end += 9

  hitter_lastname_lookup = '|'.join(top_order_list)
  hitter_lineup_df = hitter_lineup_df[hitter_lineup_df['batter_name'].str.contains(hitter_lastname_lookup)]
  hitter_lineup_df = hitter_lineup_df.dropna()

  return hitter_lineup_df

def get_list_of_hitters(position):
  if position == 'C':
    position = 'C '
  starting_lineup_df = data.get_starting_lineup()
  hitters_starting_lineup = starting_lineup_df[starting_lineup_df['Starting Lineup'].str.contains(position)]

  return list(hitters_starting_lineup['Starting Lineup'])

def generate_stack_lineup(hitter_lineup_df, starting_lineup_df, salary_data_df):
  odds_title = 'LINE|O/U'
  odds_df = starting_lineup_df[starting_lineup_df['Starting Lineup'].str.contains(odds_title)]
  odds_list = list(odds_df['Starting Lineup'])

  winner_odds = []
  over_and_under_odds = []

  for lines in odds_list:
    details_array = lines.split()
    headline = details_array[0]

    if headline == 'LINE':
      winner_odds.append(details_array[1])
    if headline == 'O/U':
      over_and_under_odds.append(float(details_array[1]))

  odds_details = {
    'favorite_to_win': winner_odds,
    'over_and_under': over_and_under_odds
  }

  odds_details_df = pd.DataFrame(odds_details)
  average_over_under = odds_details_df['over_and_under'].mean()
  odds_details_df = odds_details_df[odds_details_df['over_and_under'] > average_over_under]

  game_matchups = get_list_of_team_game_matchups(salary_data_df)
  print(f"Game matches: {game_matchups}")
  teams = list(odds_details_df['favorite_to_win'])
  print(f"Teams favored to win: {teams}")
  team_list = []
  home_team = []
  away_team = []

  for home in game_matchups:
    home_team.append(home.get('home_team'))

  for away in game_matchups:
    away_team.append(away.get('away_team'))

  game_matchups_data = {
    'home_team': home_team,
    'away_team': away_team
  }

  game_matchups_df = pd.DataFrame(game_matchups_data)

  print(f"Today's game: {game_matchups_df}")

  matchup_lookup = '|'.join(teams)
  # for team in teams:
  #   if team in game_matchups_df:
  #     team_list.append(game.get('away_team'))
  #   else:
  #     team_list.append(game.get('home_team'))

  game_matchups_df = game_matchups_df[(game_matchups_df['home_team'].str.contains(matchup_lookup)) | (game_matchups_df['away_team'].str.contains(matchup_lookup))]

  home_away_games = list(game_matchups_df['home_team']) + list(game_matchups_df['away_team'])
  hitter_team_lookup = '|'.join(home_away_games)

  print(f"List of home and away teams: {home_away_games}")

  print(f"Update matchups: {game_matchups_df}")

  hitter_lineup_df = hitter_lineup_df[hitter_lineup_df['batter_team'].str.contains(hitter_team_lookup)]
  print(f"New starting lineup: {hitter_lineup_df}")
  return hitter_lineup_df


def drop_pitchers(pitcher_lineup_df, batting_lineup_df, salary_data_df) -> pd.DataFrame:
  """Drop opposing pitchers.

  This function filters out pitchers that are facing opposing hitters.
  This ensures that opposing pitchers and hitters are not in the starting lineup in the same game.

  Parameters:
    pitcher_lineup_df: Data Frame containing a list of pitchers.
    batting_lineup_df: Data Frame containing a list of hitters.
    salary_data_df: Data Frame containing a list of players with their relative prices.

  Returns:
      DataFrame: Returns updated Data Frame.
  """
  game_matchups = get_list_of_team_game_matchups(salary_data_df)

  teams = set(batting_lineup_df['batting_teamabbrev'])
  seen = set()
  pitcher_matchups = list(pitcher_lineup_df['pitcher_teamabbrev'])

  for pitcher_matchup in pitcher_matchups:
    pitcher_away_team = next((away for away in game_matchups if away.get('away_team') == pitcher_matchup), None)

    if pitcher_away_team and pitcher_away_team.get('away_team') not in seen and pitcher_away_team.get('home_team') in teams:
      pitcher_lineup_df = pitcher_lineup_df.drop(pitcher_lineup_df[pitcher_lineup_df['pitcher_teamabbrev'] == pitcher_away_team.get('away_team')].index)

      seen.add(pitcher_away_team.get('away_team'))

  return pitcher_lineup_df

def apply_elite_statistical_pitcher_filters(pitcher_lineup_df) -> pd.DataFrame:
  """Gather statistical data for pitchers.

  This function generates a list of pitchers that have stats better than the national average.

  Parameters:
    pitcher_lineup_df: Data Frame containing a list of pitchers.

  Returns:
      DataFrame: Returns a Data Frame containing elite pitchers.
  """
  pitcher_national_average = stats_util.get_mlb_pitcher_national_averages()

  pitcher_lineup_df['elite_strikeout_K'] = pitcher_lineup_df['pitcher_strike_K_percent'] > pitcher_national_average['league_pitcher_k_bb_average']
  elite_pitchers_df = pitcher_lineup_df[pitcher_lineup_df['elite_strikeout_K'] == True]

  return elite_pitchers_df

def apply_elite_statistical_batting_filters(batting_lineup_df) -> pd.DataFrame:
  """Gather statistical data for hitters.

  This function generates a list of hitters that have stats better than the national average.

  Parameters:
    batting_lineup_df: Data Frame containing a list of hitters.

  Returns:
      DataFrame: Returns a Data Frame containing elite hitters.
  """
  batting_national_average = stats_util.get_mlb_batting_national_averages()

  batting_lineup_df['elite_wOBA'] = batting_lineup_df['batter_expected_xwOBA'] > batting_national_average['league_batting_wOBA_average']
  elite_hitters_df = batting_lineup_df[batting_lineup_df['elite_wOBA'] == True]

  return elite_hitters_df

def generate_optimal_lineup(salary_data_df, pitcher_lineup_df, batting_lineup_df,
                            is_pitcher_friendly_park, is_fanduel_lineup) -> list[dict]:
  """Lineup Generator.

  Builds a lineup based on the data being filtered.

  Parameters:
    salary_data_df: Data Frame containing a list of players with their relative prices.
    pitcher_lineup_df: Data Frame containing a list of pitchers.
    batting_lineup_df: Data Frame containing a list of hitters.
    is_pitcher_friendly_park: Boolean to determine ball park factors for pitchers.
    is_fanduel_lineup: Boolean to generate a lineup for Fanduel.

  Returns:
      List: A list of multiple lineups.
  """
  lineup_count = 0
  lineup_list = []

  while lineup_count < 10:
    starting_lineup = {}
    pitcher_starting_lineup_df = pitcher_lineup_df.dropna()
    batting_starting_lineup_df = batting_lineup_df.dropna()
    pitcher_indices = list(pitcher_starting_lineup_df.index)
    starting_players = set()
    player_salary = []

    if len(pitcher_indices) > 0:
      pitcher_one_idx = np.random.choice(pitcher_indices)
      starting_lineup['pitcher_one'] = pitcher_starting_lineup_df.loc[pitcher_one_idx]['name_id']

      drop_hitters_against_pitchers(salary_data_df, pitcher_one_idx, pitcher_starting_lineup_df, batting_starting_lineup_df,
                                    player_salary, pitcher_indices, is_pitcher_friendly_park)

    if is_fanduel_lineup:
      multi_position = 'C|1B'
      starting_lineup['catcher/1B'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, multi_position)
    else:
      if len(pitcher_indices) > 0:
        pitcher_two_idx = np.random.choice(pitcher_indices)
        starting_lineup['pitcher_two'] = pitcher_starting_lineup_df.loc[pitcher_two_idx]['name_id']

        drop_hitters_against_pitchers(salary_data_df, pitcher_two_idx, pitcher_starting_lineup_df, batting_starting_lineup_df,
                                      player_salary, pitcher_indices, is_pitcher_friendly_park)
      starting_lineup['catcher'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.CATCHER.value)
      starting_lineup['first_base'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.FIRSTBASE.value)
    starting_lineup['second_base'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.SECONDBASE.value)
    starting_lineup['third_base'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.THIRDBASE.value)
    starting_lineup['short_stop'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.SHORTSTOP.value)
    starting_lineup['outfielder_one'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)
    starting_lineup['outfielder_two'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)
    starting_lineup['outfielder_three'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, roster.PositionType.OUTFIELDER.value)

    # Fanduel includes an util field in their lineup structure
    if is_fanduel_lineup:
      # if len(pitcher_indices) > 0:
      #   pitcher_two_idx = np.random.choice(pitcher_indices)
      #   starting_lineup['util'] = pitcher_starting_lineup_df.loc[pitcher_two_idx]['name_id']
      #   player_salary.append(pitcher_starting_lineup_df.loc[pitcher_two_idx]['salary'])


      starting_lineup['util'] = get_starting_players(player_salary, starting_players, batting_starting_lineup_df, 'C|1B|2B|3B|SS|OF|OF|OF')


    # keys_exist = [k for k, v in starting_lineup.items() if v is None]
    salary_cap = sum(player_salary).astype(int)

    if is_fanduel_lineup:
      if salary_cap <= 35000 and salary_cap >= 34500:
        lineup_list.append(starting_lineup)
        lineup_count += 1
        starting_lineup['salary_cap'] = salary_cap

    if salary_cap <= 50000 and salary_cap >= 49500:
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

def get_starting_players(player_salary, starting_players, batting_starting_lineup_df, position_type):
  """Generate starting players.

  Generates starting players.

  Parameters:
    player_salary: Data Frame containing a list of players with their relative prices.
    starting_players: A list of starters
    batting_starting_lineup_df: Data Frame containing a list of hitters.
    position_type: A player's position type.

  Returns:
      DataFrame: A Data Frame of pitchers and hitters who are considered starters.
  """
  starters = batting_starting_lineup_df[batting_starting_lineup_df['position'].str.contains(position_type)]
  indices = list(starters.index)
  if len(indices) > 0:
    idx = np.random.choice(indices)
    player = starters.loc[idx]['name_id']
    while player in starting_players:
      idx = np.random.choice(indices)
      player = starters.loc[idx]['name_id']
    starting_players.add(player)
    player_salary.append(batting_starting_lineup_df.loc[idx]['salary'])

    return player

def generate_starting_lineup(salary_data_df, lineup_df, elite_players, column_name, lineup_column_name) -> pd.DataFrame:
  """Generate starting pitchers.

  Return pitchers that have the best chance for an elite performance.

  Parameters:
    salary_data_df: Data Frame containing a list of players with their relative prices.
    pitcher_lineup_df: Data Frame containing a list of pitchers.
    elite_pitchers: High performing pitchers.

  Returns:
      DataFrame: A Data Frame of pitchers and hitters who are considered starters.
  """
  salary_lookup = salary_data_df.set_index('Name')['Salary'].to_dict()
  position_lookup = salary_data_df.set_index('Name')['Position'].to_dict()
  name_id_lookup = salary_data_df.set_index('Name')['Name + ID'].to_dict()
  team_lookup = salary_data_df.set_index('Name')['Team'].to_dict()

  lineup_df['salary'] = elite_players[column_name].map(salary_lookup)
  lineup_df['position'] = elite_players[column_name].map(position_lookup)
  lineup_df['name_id'] = elite_players[column_name].map(name_id_lookup)
  lineup_df[lineup_column_name] = elite_players[column_name].map(team_lookup)
  lineup_df = lineup_df.dropna()

  return lineup_df

def drop_hitters_against_pitchers(salary_data_df, pitcher_idx, pitcher_lineup_df, batting_lineup_df,
                                  player_salary, pitcher_indices, is_pitcher_friendly_park) -> list[int]:
  """Do not include hitters that are facing opposing pitchers.

  When constructing lineups, it makes sense not to include hitters going against the opposing pitchers.
  The goal is to create a strong lineup and positive correlation.

  Parameters:
    salary_data_df: Data Frame containing a list of players with their relative prices.
    pitcher_idx: The pitcher's index within a Data Frame.
    pitcher_lineup_df: Data Frame containing a list of pitchers.
    batting_lineup_df: Data Frame containing a list of hitters.
    player_salary: Data Frame containing a list of players with their relative prices.
    pitcher_indices: A list of pitcher indicies.
    is_pitcher_friendly_park: Determines if a park is pitcher-friendly.

  Returns:
      List: A list of updated indices.
  """
  game_matchups = get_list_of_team_game_matchups(salary_data_df)

  pitcher_matchup = pitcher_lineup_df.loc[pitcher_idx]['pitcher_teamabbrev']
  pitcher_home_team = next((home for home in game_matchups if home.get('home_team') == pitcher_matchup), None)
  pitcher_away_team = next((away for away in game_matchups if away.get('away_team') == pitcher_matchup), None)

  if is_pitcher_friendly_park:
    if pitcher_home_team:
      home_team = pitcher_home_team.get('home_team')
      filter_hitters_against_pitchers(home_team, batting_lineup_df)

    if pitcher_away_team:
      away_team = pitcher_away_team.get('away_team')
      filter_hitters_against_pitchers(away_team, batting_lineup_df)
  else:
    if pitcher_home_team:
      away_team = pitcher_home_team.get('away_team')
      filter_hitters_against_pitchers(away_team, batting_lineup_df)

    if pitcher_away_team:
      home_team = pitcher_away_team.get('home_team')
      filter_hitters_against_pitchers(home_team, batting_lineup_df)

  player_salary.append(pitcher_lineup_df.loc[pitcher_idx]['salary'])
  pitcher_lineup_df.drop(pitcher_lineup_df[pitcher_lineup_df['pitcher_teamabbrev'] == pitcher_lineup_df.loc[pitcher_idx]['pitcher_teamabbrev']].index, inplace=True)

  pitcher_indices[:] = list(pitcher_lineup_df.index)

def get_list_of_team_game_matchups(salary_data_df) -> list[dict]:
  """Get a list of team game matchups.

  Generate the scheduled games for the day.

  Parameters:
    salary_data_df: Data Frame containing a list of players with their relative prices.

  Returns:
      List: A list of game matchups.
  """
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

def filter_hitters_against_pitchers(team_matchup, batting_lineup_df):
  """Get team matchup.

  Filters hitters that based on the game matchup.

  Parameters:
    teame_matchup: The current team matchup.
    batting_lineup_df: Data Frame containing a list of hitters.

  Returns:
    void: Drops hitters that are facing the opposing team.
  """
  batting_lineup_df.drop(batting_lineup_df[batting_lineup_df['batting_teamabbrev'] == team_matchup].index, inplace=True)

def generate_statistically_weak_pitchers(pitcher_lineup_df):
  pitcher_national_average = stats_util.get_mlb_pitcher_national_averages()

  pitcher_lineup_df['high_barrel'] = pitcher_lineup_df['pitcher_barrel_percent'] > pitcher_national_average['league_pitcher_barrel_average']
  pitcher_lineup_df['low_strikeout_rate'] = pitcher_lineup_df['pitcher_strike_K_percent'] < pitcher_national_average['league_pitcher_k_bb_average']

  weak_pitchers_df = pitcher_lineup_df[pitcher_lineup_df['high_barrel'] == True]
  weak_pitchers_df = pitcher_lineup_df[pitcher_lineup_df['low_strikeout_rate'] == True]

  return weak_pitchers_df