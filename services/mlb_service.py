import re
import warnings

import pandas as pd
import numpy as np
import util.mlb_stats_data_util as stats_data_util
import util.data_util as data
from rapidfuzz import process, fuzz
from collections import Counter
from ml_model.mlb_regressor_model import get_live_mlb_predictions

# Disable the SettingWithCopy/ChainedAssignment warnings
pd.options.mode.chained_assignment = None

# Suppress all other Python/Library warnings (Deprecation, etc.)
warnings.filterwarnings('ignore')

TEAM_MAPPING = {
  'SD': 'SDP',
  'SF': 'SFG',
  'KC': 'KCR',
  'WSH': 'WSN',
  'TB': 'TBR'
  }

def generate_mlb_performance_probabilities() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Generate MLB performance probabilities.

  Returns a DataFrame containing performance probabilities for pitchers and hitters based on the latest MLB data.

  Returns:
      DataFrame: A DataFrame containing performance probabilities..
  """

  pitcher_lineup_df = stats_data_util.get_mlb_pitcher_profile()
  batting_lineup_df = stats_data_util.get_mlb_batting_profile()
  pitcher_starters_df = generate_starters('pitchers', pitcher_lineup_df)
  batter_starters_df = generate_starters('hitters', batting_lineup_df)
  unique_teams = list(dict.fromkeys(get_teams()))
  home_team = {}
  away_team = {}
  index_one = 1
  trailing_index = 0
  index_two = len(unique_teams) - 2

  while index_one < len(unique_teams):
    home_team[unique_teams[trailing_index]] = unique_teams[index_one]
    index_one += 2
    trailing_index += 2
  trailing_index = len(unique_teams) - 1
  while index_two >= 0:
    away_team[unique_teams[trailing_index]] = unique_teams[index_two]
    index_two -= 2
    trailing_index -= 2

  average_innings_pitched = pitcher_starters_df['pitcher_innings_pitched'].mean()
  pitcher_starters_df['is_starter'] = pitcher_starters_df['pitcher_innings_pitched'] >= average_innings_pitched
  is_starter = pitcher_starters_df['is_starter'] == True
  is_relief_starter = pitcher_starters_df['is_starter'] == False
  pitcher_starters_df.loc[is_starter, 'position'] = 'SP'
  pitcher_starters_df.loc[is_relief_starter, 'position'] = 'RP'
  pitcher_starters_df['opposing_team'] = pitcher_starters_df['team'].map(home_team)
  pitcher_starters_df['opposing_team'] = pitcher_starters_df['opposing_team'].fillna(pitcher_starters_df['team'].map(away_team))
  opposing_pitcher_lookup = pitcher_starters_df.set_index('team')['name'].to_dict()

  batter_starters_df['opposing_team'] = batter_starters_df['team'].map(home_team)
  batter_starters_df['opposing_team'] = batter_starters_df['opposing_team'].fillna(batter_starters_df['team'].map(away_team))
  batter_starters_df['home_team'] = batter_starters_df['team'].map(home_team)
  batter_starters_df['home_team'] = batter_starters_df['home_team'].fillna(batter_starters_df['opposing_team'].map(home_team))
  batter_starters_df['away_team'] = batter_starters_df['team'].map(away_team)
  batter_starters_df['away_team'] = batter_starters_df['away_team'].fillna(batter_starters_df['opposing_team'].map(away_team))
  batter_starters_df['opposing_pitcher'] = batter_starters_df['opposing_team'].map(opposing_pitcher_lookup)
  batter_starters_df['opposing_pitcher'] = batter_starters_df['opposing_pitcher'].fillna(batter_starters_df['opposing_team'].map(opposing_pitcher_lookup))
  batter_starters_df['team'] = batter_starters_df['team'].replace(TEAM_MAPPING)
  batting_lineup_df = get_batting_data_for_model(batter_starters_df, pitcher_starters_df)
  pitcher_lineup_df = get_pitcher_data_for_model(pitcher_starters_df, batter_starters_df)
  pitcher_performance_prob = get_live_mlb_predictions(pitcher_lineup_df, 'pitcher')
  hitter_home_run_performace_prob = get_live_mlb_predictions(batting_lineup_df, 'hitter', 'home_run_prob')
  hitter_runs_performace_prob = get_live_mlb_predictions(batting_lineup_df, 'hitter', 'runs_prob')
  print(f"Pitcher performance prob: {pitcher_performance_prob}")
  print(f"Hitter home run performance prob: {hitter_home_run_performace_prob}")
  print(f"Hitter runs performance prob: {hitter_runs_performace_prob}")

  return pitcher_performance_prob, hitter_home_run_performace_prob, hitter_runs_performace_prob

def generate_starters(group_position_name, player_stats_df) -> pd.DataFrame:
  """Generate starting hitters and pitchers.

  Parameters:
    group_position_name: A string value to generate starters for pitchers or batters ('pitcher' or 'batter').
    player_stats_df: A DataFrame containing player statistics.

  Returns:
      DataFrame: Returns starting players.
  """

  if group_position_name == 'hitters':
    hitters_list = get_list_of_starters(group_position_name)
    hitters_df = clean_starting_list(hitters_list, group_position_name)
    hitters_df['team'] = get_teams()
    return get_starters(player_stats_df, hitters_df)
  else:
    pitchers_list = get_list_of_starters(group_position_name)
    pitcher_df = clean_starting_list(pitchers_list, group_position_name)
    return get_starters(player_stats_df, pitcher_df)

def get_list_of_starters(group_position_name) -> list[str]:
  """Get the list of starting hitters and pitchers.

  Parameters:
    group_position_name: A string value to generate starters for pitchers or batters ('pitcher' or 'batter').

  Returns:
    list[str]: A list of starting players.
  """
  starting_lineup_df = data.get_starting_lineup()['Starting Lineup']
  if group_position_name == 'hitters':
    positions = 'C |1B |2B |3B |SS |CF |LF |RF |DH '
    return starting_lineup_df[starting_lineup_df.str.contains(positions)]
  else:
    return list(starting_lineup_df)

def get_teams() -> list[str]:
  """
  Get the list of teams from the starting lineup dataset.

  Returns:
    list[str]: A list of unique team abbreviations.
  """
  starting_lineup_df = data.get_starting_lineup()['Starting Lineup']
  team_array = list(starting_lineup_df)
  teams = []
  players_teams = []
  index = 0
  for team_item in team_array:
    team_name_split = team_item.split()
    if len(team_name_split) <= 1 and len(team_name_split[0]) <= 3 and team_name_split[0] != '0%':
      teams.append(team_name_split[0])
  unique_teams = list(dict.fromkeys(teams))
  unique_teams = [TEAM_MAPPING.get(team, team) for team in unique_teams]
  while index < len(unique_teams):
    count = 0
    while count < 9:
      players_teams.append(unique_teams[index])
      count += 1
    index += 1
  return players_teams

def clean_starting_list(starting_list, group_position_name) -> pd.DataFrame:
  """
  Clean the list of starting players.

  Parameters:
    starting_list: A list of starting players.
    group_position_name: A string value to generate starters for pitchers or batters ('pitcher' or 'batter').

  Returns:
    DataFrame: A DataFrame containing cleaned starting players.
  """
  unique_team = list(dict.fromkeys(get_teams()))

  starting_lineup_list = []
  lineup_order = []
  position = []
  index = 0

  for starter in starting_list:
    name_array = starter.split()
    player_name = ""

    # Skip invalid strings or elements missing the handedness tag
    if not name_array or name_array[-1] not in {'L', 'R', 'S'}:
        continue

    # Check if it's a pitcher (Length 3: ['Dylan', 'Cease', 'R'])
    if group_position_name == 'pitchers':
      if len(name_array) == 3:
        player_name = " ".join(name_array[0:-1])
        position.append('P')
        lineup_order.append(0)
        starting_lineup_list.append(player_name)
    else:
      if len(name_array) >= 4:
        # Slice out everything between the position (index 0) and the handedness (index -1)
        player_name = " ".join(name_array[1:-1])
        position.append(name_array[0])

        # Track batting order (1-9 loop)
        lineup_order.append(index + 1)
        index = (index + 1) % 9  # Resets to 0 automatically after index reaches

        starting_lineup_list.append(player_name)

  starting_roster = {
    'starting_players': starting_lineup_list,
    'lineup_order': lineup_order,
    'position': position,
    'team': ''
  }

  if group_position_name == 'pitchers':
    unique_team = list(dict.fromkeys(get_teams()))
    starting_roster['team'] = unique_team
  else:
    team = get_teams()
    starting_roster['team'] = team
  starters = pd.DataFrame(starting_roster)

  return starters

def get_starters(player_stats_df, starting_list) -> pd.DataFrame:
  """
  Get the starting players based on their statistics.

  Parameters:
    player_stats_df: A DataFrame containing player statistics.
    starting_list: A DataFrame containing the list of starting players.

  Returns:
    DataFrame: A DataFrame containing the starting players and their statistics.
  """
  cleaned_starters = []
  player_lookup = {}
  team_lookup = {}
  position_lookup = {}

  score = 0
  for player in list(starting_list['starting_players']):
    score += find_best_match(player, player_stats_df)[1]

  for player, order, team, position in zip(list(starting_list['starting_players']), list(starting_list['lineup_order']), list(starting_list['team']), list(starting_list['position'])):
    best_match = find_best_match(player, player_stats_df)[0]
    score = find_best_match(player, player_stats_df)[1]
    if score >= 80:
      cleaned_starters.append(best_match)
      player_lookup[best_match] = order
      team_lookup[best_match] = team
      position_lookup[best_match] = position

  player_stats_df = player_stats_df[player_stats_df['name'].isin(cleaned_starters)]
  player_stats_df['batting_order'] = player_stats_df['name'].map(player_lookup)
  player_stats_df['team'] = player_stats_df['name'].map(team_lookup)
  player_stats_df['position'] = player_stats_df['name'].map(position_lookup)

  return player_stats_df

def find_best_match(player, player_stats_df) -> tuple[str, float]:
  """
  Find the best match for a player in the statistics DataFrame.

  Parameters:
    player: A string containing the player's name.
    player_stats_df: A DataFrame containing player statistics.

  Returns:
    tuple[str, float]: A tuple containing the best match and score.
  """
  name_array = player.split()
  player_name = f"{name_array[0]} {name_array[1]}"
  best_match, score, idx = process.extractOne(player_name, player_stats_df['name'], scorer=fuzz.WRatio)
  p_clean = clean_and_normalize(player_name)
  m_clean = clean_and_normalize(best_match)

  p_tokens = p_clean.split()
  m_tokens = m_clean.split()

  # Handle Initials: Check if the first token is a single-letter initial
  if len(p_tokens) > 0 and len(m_tokens) > 0 and len(p_tokens[0]) == 1:
      # If the first letters of the first names DO NOT match, penalize heavily
      if p_tokens[0] != m_tokens[0][0]:
          # 'R.' vs 'Ezequiel' -> 'r' != 'e' -> Hard penalization
          score = fuzz.token_sort_ratio(p_clean, m_clean) * 0.4
          return best_match, score

  # Fallback to token_sort_ratio for standard reliable matching
  score = fuzz.token_sort_ratio(p_clean, m_clean)

  # Boost score if it's a known valid initial match (e.g., 'm busch' vs 'michael busch')
  if len(p_tokens) > 1 and len(m_tokens) > 1 and p_tokens[0] == m_tokens[0][0] and p_tokens[1] == m_tokens[1]:
      score = max(score, 90.0) # Set a high confidence floor for verified initial matches

  return best_match, score

def clean_and_normalize(name):
    """Lowercase the name and strip periods (e.g., 'M. Busch' -> 'm busch')"""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(re.compile(r'\b[a-z]\.'), lambda m: m.group(0)[0], name) # 'm.' -> 'm'
    return " ".join(name.split())

def get_vegas_odds(starting_lineup_df) -> pd.DataFrame:
  """
  Extracts and processes Vegas odds from the starting lineup DataFrame.

  Parameters:
    starting_lineup_df: A DataFrame containing the starting lineup data.

  Returns:
    DataFrame: A DataFrame containing the processed Vegas odds.
  """
  odds_title = 'LINE|O/U'
  odds_df = starting_lineup_df[starting_lineup_df['Starting Lineup'].str.contains(odds_title)]
  odds_list = list(odds_df['Starting Lineup'])

  winner_odds = []
  favorite_odds = []
  over_and_under_odds = []

  for lines in odds_list:
    details_array = lines.split()
    headline = details_array[0]

    if headline == 'LINE':
      winner_odds.append(details_array[1])
      if len(details_array) >= 3:
        favorite_odds.append(int(details_array[2]))
      else:
        favorite_odds.append(-110)
    if headline == 'O/U':
      if details_array[1] == 'ï¿½':
        over_and_under_odds.append(8.0)
      else:
        over_and_under_odds.append(float(details_array[1]))

  odds_details = {
    'favorite_to_win': winner_odds,
    'favorite_odds': favorite_odds,
    'over_and_under': over_and_under_odds
  }
  odds_details_df = pd.DataFrame(odds_details)
  odds_details_df['favorite_to_win'] = odds_details_df['favorite_to_win'].replace(TEAM_MAPPING)

  return odds_details_df

def get_batting_data_for_model(batting_lineup_df, pitching_lineup_df) -> pd.DataFrame:
  park_factors_df = data.get_ball_park_factors()
  starting_lineup_df = data.get_starting_lineup()
  odds = get_vegas_odds(starting_lineup_df)

  pitcher_stats_lookup = pitching_lineup_df.set_index('name').to_dict()
  batting_lineup_df['opposing_pitcher_stance'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_throwing_hand'])

  # Replace missing stance data with 'R' (The league majority baseline)
  batting_lineup_df['opposing_pitcher_stance'] = batting_lineup_df['opposing_pitcher_stance'].fillna('R')
  batting_lineup_df['opposing_pitcher_k_percent'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_strike_K_percent'])

  # Replace missing strike out percentage data with '.21' (The league majority baseline)
  batting_lineup_df['opposing_pitcher_k_percent'] = batting_lineup_df['opposing_pitcher_k_percent'].fillna(.21)
  batting_lineup_df['opposing_pitcher_woba'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_woba_allowed'])
  batting_lineup_df['opposing_pitcher_woba'] = batting_lineup_df['opposing_pitcher_woba'].fillna(3.15)
  batting_lineup_df['opposing_pitcher_xfip'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_xfip'])
  batting_lineup_df['opposing_pitcher_xfip'] = batting_lineup_df['opposing_pitcher_xfip'].fillna(4.15)
  batting_lineup_df['opposing_pitcher_siera'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_siera'])
  batting_lineup_df['opposing_pitcher_siera'] = batting_lineup_df['opposing_pitcher_siera'].fillna(4.10)
  batting_lineup_df['opposing_pitcher_barrel_percent'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_barrel_percent'])
  batting_lineup_df['opposing_pitcher_barrel_percent'] = batting_lineup_df['opposing_pitcher_barrel_percent'].fillna(0.045)
  batting_lineup_df['opposing_pitcher_fly_ball_percent'] = batting_lineup_df['opposing_pitcher'].map(pitcher_stats_lookup['pitcher_fly_ball_percent'])
  batting_lineup_df['opposing_pitcher_fly_ball_percent'] = batting_lineup_df['opposing_pitcher_fly_ball_percent'].fillna(0.35)
  odds_lookup = odds.set_index('favorite_to_win').to_dict()
  batting_lineup_df['game_implied_total'] = batting_lineup_df['team'].map(odds_lookup['over_and_under'])
  batting_lineup_df['favorite_odds'] = batting_lineup_df['team'].map(odds_lookup['favorite_odds'])
  batting_lineup_df['game_implied_total'] = batting_lineup_df['game_implied_total'].fillna(batting_lineup_df['opposing_team'].map(odds_lookup['over_and_under']))
  batting_lineup_df = calculate_total_splits(batting_lineup_df)
  batting_lineup_df['favorite_odds'] = batting_lineup_df['favorite_odds'].fillna(-110)

  opposing_team_total = batting_lineup_df.set_index('opposing_team')['opposing_team_implied_total'].to_dict()
  opposing_team_cleaned = {k: v for k, v in opposing_team_total.items() if pd.notnull(v)}
  favorite_team_total = batting_lineup_df.set_index('team')['team_implied_total'].to_dict()
  favorite_team_cleaned = {k: v for k, v in favorite_team_total.items() if pd.notnull(v)}
  batting_lineup_df['team_implied_total'] = batting_lineup_df['team_implied_total'].fillna(batting_lineup_df['team'].map(opposing_team_cleaned))
  batting_lineup_df['opposing_team_implied_total'] = batting_lineup_df['opposing_team_implied_total'].fillna(batting_lineup_df['opposing_team'].map(favorite_team_cleaned))
  park_factors_lookup = park_factors_df.set_index('Team')['Park Factor'].to_dict()
  batting_lineup_df['park_factor'] = batting_lineup_df['home_team'].map(park_factors_lookup)

  return batting_lineup_df

def calculate_total_splits(batting_lineup_df) -> pd.DataFrame:
  """
  Calculates the total splits for each team based on the probability distribution.

  Parameters:
    batting_lineup_df: A DataFrame containing the batting lineup data.

  Returns:
    DataFrame: A DataFrame containing the calculated total splits.
  """

  # Convert moneyline to implied win probability percentage
  batting_lineup_df['favorite_probability'] = np.abs(batting_lineup_df['favorite_odds']) / (np.abs(batting_lineup_df['favorite_odds']) + 100)

  # Generate team totals based on the probability distribution
  batting_lineup_df['team_implied_total'] = (batting_lineup_df['game_implied_total'] * batting_lineup_df['favorite_probability']).round(2)
  batting_lineup_df['opposing_team_implied_total'] = (batting_lineup_df['game_implied_total'] - batting_lineup_df['team_implied_total']).round(2)

  batting_lineup_df = batting_lineup_df.drop(columns=['favorite_probability'])

  return batting_lineup_df

def get_pitcher_data_for_model(pitcher_lineup_df, batting_lineup_df) -> pd.DataFrame:
  """
  Processes pitcher data for the machine learning model.

  Parameters:
    pitcher_lineup_df: A DataFrame containing the pitcher lineup data.
    batting_lineup_df: A DataFrame containing the batting lineup data.

  Returns:
    DataFrame: A DataFrame containing the processed pitcher data.
  """

  league_batting_stats_df = data.get_league_batting_averages()
  batting_stats_lookup = league_batting_stats_df.set_index('Team').to_dict()
  league_average_strikeout_percent = (league_batting_stats_df['K%'] * 100).mean()
  pitcher_lineup_df['opposing_strikeout'] = pitcher_lineup_df['team'].map(batting_stats_lookup['SO'])
  pitcher_lineup_df['opposing_team_strikeout_multiplier'] = pitcher_lineup_df['opposing_strikeout'] / league_average_strikeout_percent

  # Compute projected outs safely if innings data exists, otherwise use defaults later
  if 'pitcher_innings_pitched' in pitcher_lineup_df.columns:
    pitcher_lineup_df['pitcher_projected_outs'] = ((pd.to_numeric(pitcher_lineup_df['pitcher_innings_pitched'], errors='coerce') / 2) * 3).round(0)
  else:
    pitcher_lineup_df['pitcher_projected_outs'] = np.where(pitcher_lineup_df['position'] == 'SP', 15.5, 3.3)

  # Ensure strikeout rate column exists and is numeric
  if 'pitcher_k_9' in pitcher_lineup_df.columns:
    k9 = pd.to_numeric(pitcher_lineup_df['pitcher_k_9'], errors='coerce').fillna(8.5)
  else:
    k9 = 8.5

  pitcher_lineup_df['projected_matchup_strikeouts'] = (
    ((k9 / 9.0) * pitcher_lineup_df['pitcher_projected_outs']) *
    pitcher_lineup_df['opposing_team_strikeout_multiplier']
  ).round(2)

  pitcher_lineup_df['opposing_woba'] = batting_lineup_df['opposing_team'].map(batting_stats_lookup['wOBA'])
  pitcher_lineup_df['opposing_xwoba'] = batting_lineup_df['opposing_team'].map(batting_stats_lookup['xwOBA'])
  if 'team_implied_total' in batting_lineup_df.columns:
    pitcher_lineup_df['opposing_team_implied_total'] = batting_lineup_df['team_implied_total']
  else:
    pitcher_lineup_df['opposing_team_implied_total'] = 4.25
  vegas_proxy_era = 2.15 + (pitcher_lineup_df['opposing_team_implied_total'] * 0.5)
  pitcher_lineup_df['pitcher_era'] = pitcher_lineup_df['pitcher_era'].fillna(vegas_proxy_era)

  # 1. Fill missing input metrics using standard role-based baselines
  # Starters average roughly 15.5 outs; Relievers average roughly 3.3 outs
  if 'pitcher_projected_outs' in pitcher_lineup_df.columns:
      pitcher_lineup_df.loc[(pitcher_lineup_df['pitcher_projected_outs'].isnull()) & (pitcher_lineup_df['position'] == 'SP'), 'pitcher_projected_outs'] = 15.5
      pitcher_lineup_df.loc[(pitcher_lineup_df['pitcher_projected_outs'].isnull()) & (pitcher_lineup_df['position'] == 'RP'), 'pitcher_projected_outs'] = 3.3
      pitcher_lineup_df['pitcher_projected_outs'] = pitcher_lineup_df['pitcher_projected_outs'].fillna(15.5)
  else:
      # Fallback if the column is entirely missing from your scraper dataset
      pitcher_lineup_df['pitcher_projected_outs'] = np.where(pitcher_lineup_df['position'] == 'SP', 15.5, 3.3)

  # 2. Fill missing Vegas Implied Totals with a standard baseline (e.g., 4.25 runs)
  if 'opposing_team_implied_total' in pitcher_lineup_df.columns:
      pitcher_lineup_df['opposing_team_implied_total'] = pitcher_lineup_df['opposing_team_implied_total'].fillna(4.25)
  else:
      pitcher_lineup_df['opposing_team_implied_total'] = 4.25

  # 3. Defensive Division Check: Avoid division by zero if a team total is somehow parsed as 0
  # Replacing 0 with a tiny epsilon value (0.01) keeps the code from crashing or producing infinite limits
  safe_denominator = np.where(pitcher_lineup_df['opposing_team_implied_total'] == 0, 0.01, pitcher_lineup_df['opposing_team_implied_total'])

  # 4. Compute the target feature cleanly
  pitcher_lineup_df['pitcher_dominance_index'] = (
      pitcher_lineup_df['pitcher_projected_outs'] / safe_denominator
  ).astype(float).round(2)

  # 5. Final validation catch-all to guarantee your Random Forest doesn't get broken values
  pitcher_lineup_df['pitcher_dominance_index'] = pitcher_lineup_df['pitcher_dominance_index'].fillna(3.65)
  pitcher_lineup_df = clean_skewed_pitcher_metrics(pitcher_lineup_df)
  pitcher_lineup_df = handle_pitcher_null_metrics(pitcher_lineup_df)

  return pitcher_lineup_df

def clean_skewed_pitcher_metrics(df) -> pd.DataFrame:
  """
  Identifies, normalizes, and hard-clamps exploded, skewed pitcher metrics
  back to standard, real-world MLB operational scales.

  Parameters:
    df: A DataFrame containing pitcher metrics.

  Returns:
    DataFrame: A DataFrame with cleaned and normalized pitcher metrics.
  """
  clean_df = df.copy()

  # 1. FIX: Normalize 'projected_matchup_strikeouts'
  # Real-world starters average between 3.5 and 8.5 strikeouts per game.
  # If the values are in the thousands, an expm1 loop leaked or multiplied raw scales.
  if 'projected_matchup_strikeouts' in clean_df.columns:
    clean_df['projected_matchup_strikeouts'] = pd.to_numeric(clean_df['projected_matchup_strikeouts'], errors='coerce')

    # Inversion rule: If an unexpected expm1 leakage occurred, apply log1p to reverse it
    clean_df['projected_matchup_strikeouts'] = np.where(
        clean_df['projected_matchup_strikeouts'] > 100,
        np.log1p(clean_df['projected_matchup_strikeouts']),
        clean_df['projected_matchup_strikeouts']
    )

    # Math Fallback check: If it's still outside standard parameters, compute from pitcher K/9 baseline
    # Standard formula: (Pitcher K/9 / 9) * 5.1 Inning Pitched average
    baseline_k_projection = (clean_df['pitcher_k_9'].fillna(8.5) / 9.0) * 5.1
    clean_df['projected_matchup_strikeouts'] = np.where(
        (clean_df['projected_matchup_strikeouts'] > 12.0) | (clean_df['projected_matchup_strikeouts'] <= 0),
        baseline_k_projection,
        clean_df['projected_matchup_strikeouts']
    )
    # Final safety clamp
    clean_df['projected_matchup_strikeouts'] = clean_df['projected_matchup_strikeouts'].clip(1.5, 11.5)

  # 2. FIX: Normalize 'opposing_team_strikeout_multiplier'
  # A multiplier feature should sit squarely on a decimal scale centered around 1.0 (e.g., 0.75 to 1.35)
  if 'opposing_team_strikeout_multiplier' in clean_df.columns:
      clean_df['opposing_team_strikeout_multiplier'] = pd.to_numeric(clean_df['opposing_team_strikeout_multiplier'], errors='coerce')

      # If it leaked exponentially, apply the inverse log1p transformation step
      clean_df['opposing_team_strikeout_multiplier'] = np.where(
          clean_df['opposing_team_strikeout_multiplier'] > 5.0,
          np.log1p(clean_df['opposing_team_strikeout_multiplier']),
          clean_df['opposing_team_strikeout_multiplier']
      )

      # Scale back to a clean ratio if it remains skewed
      clean_df['opposing_team_strikeout_multiplier'] = np.where(
          clean_df['opposing_team_strikeout_multiplier'] > 2.0,
          clean_df['opposing_team_strikeout_multiplier'] / clean_df['opposing_team_strikeout_multiplier'].mean(),
          clean_df['opposing_team_strikeout_multiplier']
      )
      # Strict structural bounding clamp
      clean_df['opposing_team_strikeout_multiplier'] = clean_df['opposing_team_strikeout_multiplier'].clip(0.70, 1.40).fillna(1.0)

  # 3. FIX: Handle Pitcher Dominance Index (Verify it maps between 0.50 and 5.50)
  if 'pitcher_dominance_index' in clean_df.columns:
    clean_df['pitcher_dominance_index'] = pd.to_numeric(clean_df['pitcher_dominance_index'], errors='coerce')
    clean_df['pitcher_dominance_index'] = np.where(
        clean_df['pitcher_dominance_index'] > 15.0,
        clean_df['pitcher_projected_outs'].fillna(16.0) / clean_df['opposing_team_implied_total'].fillna(4.25),
        clean_df['pitcher_dominance_index']
    )
    clean_df['pitcher_dominance_index'] = clean_df['pitcher_dominance_index'].clip(0.50, 6.00).fillna(3.65)

  return clean_df

def handle_pitcher_null_metrics(df) -> pd.DataFrame:
  """
  Safely finds and fills every remaining NaN across your pitcher metrics,
  varying the fallbacks by player role (SP vs RP) to keep projections accurate.

  Parameters:
    df: A DataFrame containing pitcher metrics.

  Returns:
    DataFrame: A DataFrame with NaN values filled based on role-specific baselines.
  """
  clean_df = df.copy()

  # 1. Force all feature columns to be explicitly numeric
  all_pitcher_features = [
    'pitcher_dominance_index', 'opposing_team_implied_total', 'opposing_xwoba',
    'pitcher_era', 'projected_matchup_strikeouts', 'opposing_woba',
    'pitcher_k_9', 'current_form', 'pitcher_projected_outs'
  ]

  for col in all_pitcher_features:
    if col in clean_df.columns:
      clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

  # 2. Assign dynamic, context-aware baselines for missing values
  # Starting Pitchers face more batters; Relievers throw with higher efficiency
  sp_mask = clean_df['position'].str.upper().isin(['SP', 'P'])
  rp_mask = clean_df['position'].str.upper().isin(['RP'])

  # --- Feature 1: Innings Pitched / Volume Context ---
  if 'pitcher_projected_outs' in clean_df.columns:
    clean_df.loc[sp_mask & clean_df['pitcher_projected_outs'].isnull(), 'pitcher_projected_outs'] = 5.1
    clean_df.loc[rp_mask & clean_df['pitcher_projected_outs'].isnull(), 'pitcher_projected_outs'] = 1.1
    clean_df['pitcher_projected_outs'] = clean_df['pitcher_projected_outs'].fillna(5.1)

  # --- Feature 2: Core Efficiency Metrics ---
  if 'pitcher_era' in clean_df.columns:
    clean_df.loc[sp_mask & clean_df['pitcher_era'].isnull(), 'pitcher_era'] = 4.15
    clean_df.loc[rp_mask & clean_df['pitcher_era'].isnull(), 'pitcher_era'] = 3.85
    clean_df['pitcher_era'] = clean_df['pitcher_era'].fillna(4.15)

  if 'pitcher_k_9' in clean_df.columns:
    clean_df.loc[sp_mask & clean_df['pitcher_k_9'].isnull(), 'pitcher_k_9'] = 8.50
    clean_df.loc[rp_mask & clean_df['pitcher_k_9'].isnull(), 'pitcher_k_9'] = 9.20
    clean_df['pitcher_k_9'] = clean_df['pitcher_k_9'].fillna(8.50)

  # --- Feature 3: Matchup & Opposing Lineup Context ---
  if 'opposing_team_implied_total' in clean_df.columns:
    clean_df['opposing_team_implied_total'] = clean_df['opposing_team_implied_total'].fillna(4.25)

  if 'opposing_woba' in clean_df.columns:
    clean_df['opposing_woba'] = clean_df['opposing_woba'].fillna(0.315)

  if 'opposing_xwoba' in clean_df.columns:
      clean_df['opposing_xwoba'] = clean_df['opposing_xwoba'].fillna(0.320)

  # --- Feature 5: Composite Structural Indices ---
  if 'pitcher_dominance_index' in clean_df.columns:
    # Calculate index using the safe volume metrics we just imputed
    calculated_dominance = (clean_df['pitcher_projected_outs'] * 3.0) / clean_df['opposing_team_implied_total']
    clean_df['pitcher_dominance_index'] = clean_df['pitcher_dominance_index'].fillna(calculated_dominance)
    clean_df['pitcher_dominance_index'] = clean_df['pitcher_dominance_index'].fillna(3.65)

  if 'projected_matchup_strikeouts' in clean_df.columns:
    calculated_matchup_k = (clean_df['pitcher_k_9'] / 9.0) * clean_df['pitcher_projected_outs']
    clean_df['projected_matchup_strikeouts'] = clean_df['projected_matchup_strikeouts'].fillna(calculated_matchup_k)
    clean_df['projected_matchup_strikeouts'] = clean_df['projected_matchup_strikeouts'].fillna(4.50)

  return clean_df

def process_batter_data_model(processed_df) -> pd.DataFrame:
  """Process data for model input.

  Parameters:
    processed_df: Data Frame containing a list of hitters.

  Returns:
    DataFrame: A DataFrame containing the processed batter data for model input.
 """

  league_averages = {
    'batter_actual_wOBA': 0.315,
    'batter_expected_xwOBA': 0.315,
    'batter_ISO': 0.160,
    'batter_barrel_percent': 7.5,
    'batting_line_drive_rate': 21.0,
    'opposing_pitcher_k_percent': 0.220,
    'opposing_pitcher_fly_ball_percent': 0.33,
    'opposing_pitcher_xfip': 0.214,
    'platoon_vs_rhp': 0.301,
    'platoon_vs_lhp': 0.330,
    'opposing_pitcher_woba': 0.315,
    'opposing_pitcher_siera': 4.10,
    'park_factor': 100.0,
    'batting_order': 8.0,
    'favorite_odds': -110.0,
    'game_implied_total': 8.5,
    'team_implied_total': 4.25
  }
  processed_df['data_was_imputed'] = 0
  # Mark rows where opposing pitcher metrics were missing (if columns exist)
  if 'opposing_pitcher_woba' in processed_df.columns:
    processed_df.loc[processed_df['opposing_pitcher_woba'].isnull(), 'data_was_imputed'] = 1
  if 'opposing_pitcher_siera' in processed_df.columns:
    processed_df.loc[processed_df['opposing_pitcher_siera'].isnull(), 'data_was_imputed'] = 1

  # Loop through and fill missing values with specific baseline dictionary
  for col, fallback_value in league_averages.items():
    if col not in processed_df.columns:
      processed_df[col] = fallback_value
    processed_df[col] = processed_df[col].fillna(fallback_value)
    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(fallback_value)

  # Convert American Moneyline to clean Implied Win Probability (0.0 to 1.0)
    # Handles calculation differences for favorites (-) vs underdogs (+)
    if 'favorite_odds' in processed_df.columns:
        processed_df['team_win_probability'] = np.where(
            processed_df['favorite_odds'] < 0,
            abs(processed_df['favorite_odds']) / (abs(processed_df['favorite_odds']) + 100),
            100 / (processed_df['favorite_odds'] + 100)
        )
    else:
        processed_df['team_win_probability'] = 0.50

    # Extract Exact Team Implied Runs (Isolates heavy favorites in high-scoring games)
    # Shifts a share of the game total to the favorite based on win percentage
    if 'game_implied_total' in processed_df.columns:
        margin_shifter = (processed_df['team_win_probability'] - 0.5) * 1.5
        processed_df['team_implied_runs'] = (processed_df['game_implied_total'] / 2) + margin_shifter
    else:
        processed_df['team_implied_runs'] = 4.25

    # Binary Stacking Triggers (Flag the absolute best game environments on the slate)
    # Single-slate logic: Treats the entire DataFrame as tonight's unique slate
    # Finds the highest game total active tonight and flags it dynamically
    # Determine the slate max safely
    if 'game_implied_total' in processed_df.columns:
      slate_max = processed_df['game_implied_total'].max()
    else:
      slate_max = np.nan

    # Safety check: ensure slate_max isn't NaN or empty
    if pd.notnull(slate_max) and 'game_implied_total' in processed_df.columns:
        processed_df['is_highest_total_on_slate'] = (processed_df['game_implied_total'] == slate_max).astype(int)
    else:
        # Fallback behavior when column missing
        processed_df['is_highest_total_on_slate'] = 0

    # Heavy Favorite Flag (True if win chance is >= 58%, roughly equivalent to a -140 moneyline)
    processed_df['is_heavy_favorite'] = (processed_df['team_win_probability'] >= 0.58).astype(int)

  # This captures the non-linear drop-off in opportunity volume
  pa_map = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.2, 5: 4.1, 6: 3.8, 7: 3.4, 8: 3.3, 9: 3.0}
  processed_df['expected_plate_appearances'] = processed_df['batting_order'].map(pa_map).fillna(3.0)

  # This turns abstract efficiency into concrete projected counting opportunity
  processed_df['projected_volume_woba'] = processed_df['batter_actual_wOBA'] * processed_df['expected_plate_appearances']
  processed_df['projected_volume_xwoba'] = processed_df['batter_expected_xwOBA'] * processed_df['expected_plate_appearances']
  processed_df['projected_volume_iso'] = processed_df['batter_ISO'] * processed_df['expected_plate_appearances']
  processed_df['projected_volume_barrel'] = (processed_df['batter_barrel_percent'] / 100) * processed_df['expected_plate_appearances']
  processed_df['projected_volume_linedrive'] = processed_df['batting_line_drive_rate'] * processed_df['expected_plate_appearances']
  processed_df['lineup_turnover_factor'] = (
    processed_df['expected_plate_appearances'] * (processed_df['team_implied_total'] / 4.5)
  )
  processed_df['air_clash_factor'] = processed_df['batter_barrel_percent'] * processed_df['opposing_pitcher_fly_ball_percent']
  processed_df['power_modifier'] = 1.0 + processed_df['projected_volume_iso']

  # Weighting the platoon advantage by the amount of game time they will actually see
  processed_df['platoon_pa_rhp'] = processed_df['platoon_vs_rhp'].fillna(0)
  processed_df['platoon_pa_lhp'] = processed_df['platoon_vs_lhp'].fillna(0)

  processed_df['platoon_factor'] = processed_df['platoon_pa_rhp'] + processed_df['platoon_pa_lhp']

  # ===== ENHANCED PITCHER VULNERABILITY SCORING =====
  # Multi-factor pitcher weakness index normalized to league averages
  # Higher values = more vulnerable pitcher = higher hitter upside

  # Normalize wOBA to league average (0.315)
  processed_df['woba_vulnerability'] = (
      (processed_df['opposing_pitcher_woba'] / 0.315) - 1.0
  ) * 0.35  # 35% weight, scaled relative to baseline

  # Normalize SIERA to league average (4.1)
  processed_df['siera_vulnerability'] = (
      (processed_df['opposing_pitcher_siera'] / 4.1) - 1.0
  ) * 0.35  # 35% weight, scaled relative to baseline

  # K% and BB/9 impact (higher K% = fewer hits, lower BB% = fewer free passes)
  # Use available metrics or defaults
  processed_df['pitcher_k_bb_vulnerability'] = (
      (0.22 - processed_df['opposing_pitcher_k_percent'].fillna(0.22)) / 0.22
  ) * 0.20  # 20% weight: lower K% = more vulnerable

  # xFIP normalization (higher = worse pitcher)
  processed_df['xfip_vulnerability'] = (
      (processed_df['opposing_pitcher_xfip'].fillna(4.15) / 4.15) - 1.0
  ) * 0.10  # 10% weight

  # Combined pitcher target index (baseline 1.0 for league average pitcher)
  processed_df['pitcher_target_index'] = (
      1.0 +
      processed_df['woba_vulnerability'] +
      processed_df['siera_vulnerability'] +
      processed_df['pitcher_k_bb_vulnerability'] +
      processed_df['xfip_vulnerability']
  )

  # Ceiling multiplier based on pitcher weakness (more aggressive scaling)
  # Poor pitchers (index > 1.15) get 1.3x ceiling boost
  # Good pitchers (index < 0.85) get 0.8x ceiling dampener
  processed_df['pitcher_weakness_ceiling_multiplier'] = (
      0.8 + (processed_df['pitcher_target_index'] * 0.25)
  ).clip(lower=0.75, upper=1.45)

  # Best hitters facing the most vulnerable pitchers with the most volume
  processed_df['matchup_score'] = processed_df['projected_volume_xwoba'] * processed_df['pitcher_target_index']

  # Scale the matchup data by the game environment (Vegas totals + Park Factors)
  processed_df['env_multiplier'] = (processed_df['game_implied_total'] / 4.5) * (processed_df['park_factor'] / 100)
  processed_df['final_weighted_projection_signal'] = processed_df['matchup_score'] * processed_df['env_multiplier']

  processed_df = simulate_slate_game_outcomes(processed_df)

  return processed_df

def simulate_slate_game_outcomes(processed_df, simulations=1000, random_state=42) -> pd.DataFrame:
  """
  Runs play-by-play slate simulations to extract structural distribution
  features (Ceilings, Volatility, Blowout Potential) for the Random Forest.

  Parameters:
    processed_df: A DataFrame containing the processed batter data.
    simulations: Number of Monte Carlo simulations to run per game.
    random_state: Seed for reproducibility of random simulations.

  Returns:
    DataFrame: A DataFrame with additional simulation-based features added.
  """
  if processed_df.empty:
      return processed_df

  # Generate unique game tracking identifiers if missing
  if 'game_id' not in processed_df.columns:
      processed_df['game_id'] = processed_df.apply(
          lambda r: "_".join(np.sort([r['team'], r['opposing_team']])), axis=1
      )

  # Group to get a clear picture of the baseline game environments
  game_summary = processed_df.groupby('game_id').agg(
      game_implied_total=('game_implied_total', 'mean'),
      avg_pitcher_vuln=('pitcher_target_index', 'mean'),
      avg_park_factor=('park_factor', 'mean')
  ).reset_index()

  rng = np.random.default_rng(random_state)

  # Storage maps for our advanced distribution features
  game_90th_ceiling_map = {}
  game_volatility_map = {}
  game_blowout_prob_map = {}

  for _, row in game_summary.iterrows():
    mean_total = float(row['game_implied_total']) if pd.notnull(row['game_implied_total']) else 8.5
    std_total = max(1.0, mean_total * 0.15)
    vulnerability_boost = 1.0 + ((float(row['avg_pitcher_vuln']) - 1.0) * 0.20) if pd.notnull(row['avg_pitcher_vuln']) else 1.0
    park_boost = (float(row['avg_park_factor']) / 100.0) if pd.notnull(row['avg_park_factor']) else 1.0

    # Simulate using a Gamma distribution to capture the realistic right-hand tail of baseball scores
    # (Baseball scores cannot go below 0, but can explode up to 15+ runs)
    shape = (mean_total / std_total) ** 2
    scale = (std_total ** 2) / mean_total

    raw_outcomes = rng.gamma(shape=shape, scale=scale, size=simulations)
    adjusted_outcomes = raw_outcomes * vulnerability_boost * park_boost

    # --- EXTRACT STRUCTURAL FEATURES ---
    # 1. 90th Percentile: What does this game look like when the offenses explode?
    game_90th_ceiling_map[row['game_id']] = np.percentile(adjusted_outcomes, 90)

    # 2. Volatility: How unpredictable is this specific game environment?
    game_volatility_map[row['game_id']] = np.std(adjusted_outcomes)

    # 3. Blowout Probability: What is the mathematical likelihood of this game crossing 10.5 total runs?
    game_blowout_prob_map[row['game_id']] = np.mean(adjusted_outcomes > 10.5)

  # Map the advanced features back to the main player DataFrame
  processed_df['sim_game_90th_ceiling'] = processed_df['game_id'].map(game_90th_ceiling_map).fillna(12.0)
  processed_df['sim_game_volatility'] = processed_df['game_id'].map(game_volatility_map).fillna(2.5)
  processed_df['sim_game_blowout_probability'] = processed_df['game_id'].map(game_blowout_prob_map).fillna(0.20)

  # Backward compatible simulation signals for legacy projection calculations
  processed_df['simulated_game_strength_index'] = 1.0
  if 'avg_pitcher_vuln' in game_summary.columns and 'game_id' in processed_df.columns:
    game_strength_map = {
      row['game_id']: np.clip(
          1.0 + ((float(row['avg_pitcher_vuln']) - 1.0) * 0.20),
          0.82,
          1.25
      )
      for _, row in game_summary.iterrows()
    }
    processed_df['simulated_game_strength_index'] = processed_df['game_id'].map(game_strength_map).fillna(1.0)

  processed_df['simulated_game_outcome_multiplier'] = np.clip(
      1.0 + (processed_df['simulated_game_strength_index'] - 1.0) * 0.20,
      0.82,
      1.30
  )
  processed_df['simulated_game_weakness_multiplier'] = np.clip(
      np.where(
          processed_df['simulated_game_strength_index'] < 1.0,
          1.0 - ((1.0 - processed_df['simulated_game_strength_index']) * 0.18),
          1.0
      ),
      0.82,
      1.0
  )

  # Combine metrics into a single, high-signal multiplier for the final feature array
  processed_df['simulated_stack_leverage_index'] = (
      (processed_df['sim_game_90th_ceiling'] / 8.5) *
      (1.0 + processed_df['sim_game_blowout_probability'])
  )

  # This creates a high-scale metric that heavily values power hitters in volatile games
  projected_volume_iso = processed_df['projected_volume_iso'] if 'projected_volume_iso' in processed_df.columns else 0.0
  projected_volume_xwoba = processed_df['projected_volume_xwoba'] if 'projected_volume_xwoba' in processed_df.columns else 0.0
  projected_volume_barrel = processed_df['projected_volume_barrel'] if 'projected_volume_barrel' in processed_df.columns else 0.0

  processed_df['sim_scaled_volume_iso'] = (
    projected_volume_iso * processed_df['simulated_stack_leverage_index']
  )

  # Scale expected offensive production by the 90th percentile game ceiling
  processed_df['sim_scaled_volume_xwoba'] = (
    projected_volume_xwoba * (processed_df['sim_game_90th_ceiling'] / 8.5)
  )

  # Create an explosive "Homerun Stack Factor"
  processed_df['sim_scaled_barrel_leverage'] = (
    projected_volume_barrel * processed_df['sim_game_blowout_probability'] * 10
  )

  # Calculate the average wOBA volume of the player's specific team tonight
  if 'projected_volume_woba' in processed_df.columns:
    processed_df['team_avg_volume_woba'] = processed_df.groupby('team')['projected_volume_woba'].transform('mean')
    # Isolate the player's relative difference from their team baseline
    processed_df['relative_player_volume_delta'] = (
      processed_df['projected_volume_woba'] - processed_df['team_avg_volume_woba']
    )
  else:
    processed_df['team_avg_volume_woba'] = 0.0
    processed_df['relative_player_volume_delta'] = 0.0

  return processed_df