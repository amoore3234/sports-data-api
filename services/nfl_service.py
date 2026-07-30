
from typing import Tuple
import numpy as np
import pandas as pd
import json
import nflreadpy as nfl
from ml_model.nfl_predictive_model import get_nfl_player_baselines

def get_nfl_team_offense_stats() -> json:
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

def get_nfl_team_defense_stats() -> json:
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

def get_overall_weighted_defensive_average() -> float:
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

def get_nfl_player_snap_count() -> json:
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

def get_nfl_odds() -> json:
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

def get_nfl_teams() -> json:
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

def generate_nfl_performance_probabilities() -> pd.DataFrame:
  """Generates NFL player performance probabilities based on historical data and simulations.

  Returns:
    pd.DataFrame: A dataframe containing player profiles with simulation results.
  """

  nfl_players_baselines_df = build_nflverse_feature_matrix()
  nfl_players_baselines_df['player'] = nfl_players_baselines_df['player'].str.lower()

  nfl_players_performance = get_nfl_player_baselines(nfl_players_baselines_df)
  nfl_players_predictions = nfl_game_simulation(nfl_players_performance, num_sims=5000)

  print("\n--- TOP 10 QUARTERBACK PASSING PROFILES ---")
  df_qbs = nfl_players_predictions.query("Position == 'QB'").sort_values(by='Prob_Over_300_Pass_Yds_%', ascending=False)
  print(df_qbs[[
    'Player_Name', 'Team', 'Opponent', 'Prob_Over_300_Pass_Yds_%', 'Expected_Touchdowns', 'High_Volume_Probability_%'
  ]].head(10).to_string(index=False))

  print("\n--- TOP 10 RUNNING BACK RUSHING PROFILES ---")
  df_rbs = nfl_players_predictions.query("Position == 'RB'").sort_values(by='Prob_Over_100_Rush_Yds_%', ascending=False)
  print(df_rbs[[
    'Player_Name', 'Team', 'Opponent', 'Prob_Over_100_Rush_Yds_%', 'Expected_Touchdowns', 'High_Volume_Probability_%'
  ]].head(10).to_string(index=False))

  print("\n--- TOP 10 WIDE RECEIVER RECEIVING PROFILES ---")
  df_wrs = nfl_players_predictions.query("Position == 'WR'").sort_values(by='High_Volume_Probability_%', ascending=False)
  print(df_wrs[[
    'Player_Name', 'Team', 'Opponent', 'Prob_Over_100_Rec_Yds_%', 'Expected_Touchdowns', 'High_Volume_Probability_%'
  ]].head(10).to_string(index=False))

  return nfl_players_predictions

def transform_to_short_name(full_name_string) -> str:
  """
  Converts 'First Last' into 'f last' (e.g., 'Drake Maye' -> 'd maye').
  Handles trailing whitespace and case variations automatically.

  Args:
    full_name_string (str): The full name of the player.

  Returns:
    str: The transformed short name in the format 'f last'.
  """
  if pd.isna(full_name_string) or not isinstance(full_name_string, str):
    return "missing_name"

  # Split the name into components based on spaces
  name_parts = full_name_string.strip().split()

  if len(name_parts) < 2:
    # Fallback if only a single name exists
    return full_name_string.lower().strip()

  # List common suffixes to ignore when locating the true last name
  suffixes_to_ignore = ['jr', 'sr', 'ii', 'iii', 'iv', 'v', 'esq']

  # If the last word is a suffix, pop it out or step back one index slot
  if name_parts[-1] in suffixes_to_ignore:
    last_name = name_parts[-2]
  else:
    last_name = name_parts[-1]

  first_initial = name_parts[0][0].lower()
  # Grabs the last element to safely bypass middle names
  last_name = name_parts[-1].lower()

  return f"{first_initial}.{last_name}"

def build_nflverse_feature_matrix() -> pd.DataFrame:
  """
  Downloads raw play-by-play data from nflverse, parses targets and air yards,
  and returns a structured dataframe matching your model features.

  Returns:
    pd.DataFrame: A dataframe containing player-level features for model input.
  """
  # Load player game-level stats for multiple seasons
  player_stats = nfl.load_player_stats(2025)
  player_stats = player_stats.to_pandas()
  redzone_players_df, redzone_team_df = build_player_redzone_metrics()

  epa_columns = ['passing_epa', 'rushing_epa', 'receiving_epa', 'targets', 'carries']

  player_stats[epa_columns] = player_stats[epa_columns].fillna(0.0)
  player_stats['actual_game_total_tds'] = player_stats['rushing_tds'] + player_stats['receiving_tds']

  player_stats['total_volume_opportunities'] = np.where(
    player_stats['position'] == 'RB',
    player_stats['carries'],
    player_stats['targets']
  )

  # Compute their historical touchdown conversion rate PER SINGLE TOUCH
  player_stats['historical_td_prob_per_opportunity'] = np.where(
    player_stats['total_volume_opportunities'] > 0,
    player_stats['actual_game_total_tds'] / player_stats['total_volume_opportunities'],
    0.0
  )

  model_features_df = player_stats.groupby(['season', 'player_name', 'player_id', 'team']).agg(
    avg_target_share=('target_share', 'mean'),       # Found directly in dictionary
    avg_air_yards_share=('air_yards_share', 'mean'), # Found directly in dictionary
    recent_targets=('targets', 'mean'),
    receptions=('receptions', 'mean'),
    avg_receiving_yards=('receiving_yards', 'mean'),
    avg_receiving_epa=('receiving_epa', 'mean'),
    avg_carries=('carries', 'mean'),
    avg_rushing_yards=('rushing_yards', 'mean'),
    avg_rushing_epa=('rushing_epa', 'mean'),
    avg_passing_yards=('passing_yards', 'mean'),
    avg_passing_tds=('passing_tds', 'mean'),
    avg_passing_epa=('passing_epa', 'mean'),
    actual_game_total_tds=('actual_game_total_tds', 'mean'),
    recent_pass_attempts=('attempts', 'mean'),
    historical_td_prob_per_opportunity=('historical_td_prob_per_opportunity', 'mean'),
    opponent=('opponent_team', 'first')
  ).reset_index()

  nfl_odds = get_nfl_odds()
  data_dict = json.loads(nfl_odds)
  over_under_lookup = {}

  for odds in data_dict:
    team_key = odds['team']
    over_under_lookup[team_key] = float(odds['over_under'])

  # Calculate opponent-specific metrics
  # Group by the 'opponent' column to find total targets faced and total EPA surrendered
  defense_performance_df = player_stats.groupby(['season', 'opponent_team']).agg(
    total_targets_faced=('targets', 'sum'),
    total_receiving_epa_allowed=('receiving_epa', 'sum')
  ).reset_index()

  defense_rush_performance_df = player_stats.groupby(['season', 'opponent_team']).agg(
    total_carries_faced=('carries', 'sum'),
    total_rushing_epa_allowed=('rushing_epa', 'sum')
  ).reset_index()

  # Calculate league-wide baseline: Average receiving EPA per single target
  league_baseline_epa_per_target = player_stats['receiving_epa'].sum() / player_stats['targets'].sum()
  league_baseline_epa_per_rush = player_stats['rushing_epa'].sum() / player_stats['carries'].sum()
  model_features_df['catch_rate'] = np.where(
    model_features_df['recent_targets'] > 0,
    model_features_df['receptions'] / model_features_df['recent_targets'],
    0.0
  )

  model_features_df['catch_rate'] = model_features_df['catch_rate'].round(4)

  # Calculate each team's raw average defensive EPA allowed per target
  defense_performance_df['def_epa_per_target'] = defense_performance_df['total_receiving_epa_allowed'] / defense_performance_df['total_targets_faced']
  defense_rush_performance_df['def_rushing_epa_per_carry'] = defense_rush_performance_df['total_rushing_epa_allowed'] / defense_rush_performance_df['total_carries_faced']

  # THE PROXY FORMULA: Compute the variance against the league baseline
  # Negative = Elite secondary (holds offenses below average)
  # Positive = Weak pass defense (surrenders high-efficiency plays)
  defense_performance_df['opp_pass_dvoa'] = defense_performance_df['def_epa_per_target'] - league_baseline_epa_per_target
  defense_rush_performance_df['opp_rush_dvoa'] = defense_rush_performance_df['def_rushing_epa_per_carry'] - league_baseline_epa_per_rush
  defense_performance_df['implied_team_total'] = defense_performance_df['opponent_team'].map(over_under_lookup).fillna(44.5)

  df_merged = pd.merge(
    model_features_df,
    defense_performance_df,
    left_on=['season', 'opponent'],
    right_on=['season', 'opponent_team'],
    how='left'
  )

  df_merged = pd.merge(
    df_merged,
    defense_rush_performance_df,
    left_on=['season', 'opponent'],
    right_on=['season', 'opponent_team'],
    how='left'
  )

  df_merged = pd.merge(
    df_merged,
    redzone_players_df,
    on='player_id',
    how='left'
  ).fillna(0.0)

  # Map team red zone execution rates using team abbreviations
  df_merged = pd.merge(
    df_merged,
    redzone_team_df,
    on='team',
    how='left'
  ).fillna(0.35)

  # Clean the matrix to return straight to your Random Forest input pipeline
  df_final_features = defense_performance_df[['opponent_team', 'opp_pass_dvoa']]

  df_final_features = df_merged.groupby(['player_name', 'player_id', 'team']).agg(
    avg_target_share=('avg_target_share', 'mean'),
    avg_air_yards_share=('avg_air_yards_share', 'mean'),
    recent_targets=('recent_targets', 'mean'),
    actual_game_yprr=('catch_rate', 'mean'),
    opp_pass_dvoa=('opp_pass_dvoa', 'mean'),
    opp_rush_dvoa=('opp_rush_dvoa', 'mean'),
    # TODO: Ensure to use current implied totals from starting lineups dataset.
    implied_team_total=('implied_team_total', 'mean'),
    avg_carries=('avg_carries', 'mean'),
    avg_receiving_epa=('avg_receiving_epa', 'mean'),
    avg_receiving_yards=('avg_receiving_yards', 'mean'),
    avg_rushing_yards=('avg_rushing_yards', 'mean'),
    avg_rushing_epa=('avg_rushing_epa', 'mean'),
    avg_passing_yards=('avg_passing_yards', 'mean'),
    avg_passing_tds=('avg_passing_tds', 'mean'),
    avg_passing_epa=('avg_passing_epa', 'mean'),
    actual_game_total_tds=('actual_game_total_tds', 'mean'),
    opponent=('opponent', 'first'),
    seasonal_rz_targets=('seasonal_rz_targets', 'mean'),
    seasonal_rz_carries=('seasonal_rz_carries', 'mean'),
    recent_pass_attempts=('recent_pass_attempts', 'mean'),
    team_redzone_success_rate=('team_redzone_success_rate', 'mean'),
    historical_td_prob_per_opportunity=('historical_td_prob_per_opportunity', 'mean'),
  ).reset_index()

  df_final_features.rename(columns={'player_name': 'player'}, inplace=True)

  return df_final_features

def build_player_redzone_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Downloads raw play-by-play data, isolates true red-zone target volume shares,
  and computes team conversion efficiency.

  Returns:
    pd.DataFrame: Player-level red zone metrics.
    pd.DataFrame: Team-level red zone success rates.
  """
  player_stats = nfl.load_pbp(2025)
  player_stats = player_stats.to_pandas()

  redzone_df = player_stats[(player_stats['yardline_100'] <= 20) & (player_stats['play_type'].isin(['pass', 'run']))].copy()

  # Group by the possessing team ('posteam') to find total touchdowns scored inside the 20
  team_redzone_totals = player_stats[player_stats['yardline_100'] <= 20].groupby('posteam').agg(
    total_rz_plays=('play_id', 'count'),
    total_rz_tds=('touchdown', 'sum')
  ).reset_index()

  team_redzone_totals['team_redzone_success_rate'] = team_redzone_totals['total_rz_tds'] / team_redzone_totals['total_rz_plays']
  team_rz_lookup = team_redzone_totals[['posteam', 'team_redzone_success_rate']].rename(columns={'posteam': 'team'})

  # Isolate Individual Player Red Zone Volume Shares
  rz_pass_plays = redzone_df[(redzone_df['play_type'] == 'pass') & (redzone_df['receiver_player_id'].notna())]
  df_rz_targets = rz_pass_plays.groupby('receiver_player_id').agg(
    seasonal_rz_targets=('receiver_player_id', 'count')
  ).reset_index().rename(columns={'receiver_player_id': 'player_id'})

  # Red Zone Carries & Goal-Line Efficiency (RBs/QBs)
  rz_run_plays = redzone_df[(redzone_df['play_type'] == 'run') & (redzone_df['rusher_player_id'].notna())]
  df_rz_carries = rz_run_plays.groupby('rusher_player_id').agg(
    seasonal_rz_carries=('rusher_player_id', 'count')
  ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

  # Merge player metrics into a single, non-duplicated red zone lookup table
  df_rz_merged = pd.merge(df_rz_targets, df_rz_carries, on='player_id', how='outer').fillna(0.0)

  return df_rz_merged, team_rz_lookup

def nfl_game_simulation(player_baselines_df, num_sims=5000) -> pd.DataFrame:
  """
  Ingests position-stratified predictions and explicitly uses defensive
  DVOA metrics to shift the shape parameters of your simulation distributions.

  Args:
    player_baselines_df (pd.DataFrame): Dataframe containing player baseline predictions.
    num_sims (int): Number of Monte Carlo simulations to run for each player.

  Returns:
    pd.DataFrame: Dataframe containing player profiles with simulation results.
  """
  num_players = len(player_baselines_df)
  player_profiles = []

  print(f"Executing DVOA-infused matchup simulation across {num_players} player profiles...")

  for idx in range(num_players):
    player = player_baselines_df.iloc[idx]
    pos = player['position_assigned']

    # Extract the specific defensive metrics we engineered in previous steps
    pass_dvoa = float(player.get('opp_pass_dvoa', 0.0))
    rush_dvoa = float(player.get('opp_rush_dvoa', 0.0))

    sim_volume = np.zeros(num_sims)
    sim_yards = np.zeros(num_sims)
    sim_tds = np.zeros(num_sims)

    # DVOA WORKLOAD MULTIPLIER (Volume Scaling)
    # Elite defenses reduce overall team plays.
    # Positive DVOA increases expected volume; Negative DVOA suppresses it.
    if pos == 'RB':
      # 80% elasticity on ground volume
      vol_matchup_multiplier = 1.0 + (rush_dvoa * 0.8)
    else:
      # 60% elasticity on passing volume
      vol_matchup_multiplier = 1.0 + (pass_dvoa * 0.6)

    calibrated_vol_base = player['pred_vol'] * vol_matchup_multiplier
    calibrated_vol_std = player['pred_vol_std'] * (1.0 + abs(rush_dvoa if pos == 'RB' else pass_dvoa))

    # Apply a Gaussian script shift to workload counts per simulation pass
    fuzzed_lambdas = np.random.normal(loc=calibrated_vol_base, scale=max(0.2, calibrated_vol_std), size=num_sims)
    fuzzed_lambdas = np.maximum(1.0, fuzzed_lambdas)

    for sim in range(num_sims):
      # Poisson Volume Generation
      opportunities = np.random.poisson(lam=fuzzed_lambdas[sim])
      sim_volume[sim] = opportunities

      if opportunities <= 0:
        continue

      # Shifts the distribution curve based on defensive efficiency.
      if pos == 'QB':
        completions = np.random.binomial(n=opportunities, p=0.64)
        if completions > 0:
          sigma = 0.28
          # Passing DVOA directly scales target yardage boundaries
          target_yards = max(50.0, player['pred_yds'] * (1.0 + pass_dvoa))
          mu = np.log(target_yards) - (sigma**2 / 2.0)
          sim_yards[sim] = np.random.lognormal(mean=mu, sigma=sigma)
      elif pos == 'WR':
        catches = np.random.binomial(n=opportunities, p=0.65)
        if catches > 0:
          sigma = 0.45
          # Weak secondaries inflate deep downfield chunk metrics
          target_yards = max(10.0, player['pred_yds'] * (1.0 + pass_dvoa * 1.2))
          mu = np.log(target_yards) - (sigma**2 / 2.0)
          sim_yards[sim] = np.random.lognormal(mean=mu, sigma=sigma)
      else: # Running Backs
        sigma = 0.22
        # Elite run fronts stifle yardage per carry floors
        target_yards = max(10.0, player['pred_yds'] * (1.0 + rush_dvoa))
        mu = np.log(target_yards) - (sigma**2 / 2.0)
        sim_yards[sim] = np.random.lognormal(mean=mu, sigma=sigma)

      # Alters the probability of a touch resulting in a score.
      if pos == 'QB':
        qb_td_lambda = player['pred_tds'] * (1.0 + pass_dvoa)
        sim_tds[sim] = np.random.poisson(lam=max(0.01, qb_td_lambda))
      else:
        # Skill Position Touchdown Prob: Combines team totals, individual
        # red-zone usage rates, and the opponent's defensive DVOA metrics
        base_prob = player['pred_tds'] * (player['implied_team_total'] / 24.0)

        # Apply defensive resistance filters to the conversion percentages
        if pos == 'RB':
          dynamic_prob = base_prob * (1.0 + rush_dvoa * 1.5) # High touchdown sensitivity to run defense
        else:
          dynamic_prob = base_prob * (1.0 + pass_dvoa * 1.2)

        dynamic_prob = np.clip(dynamic_prob, 0.001, 0.12)

        scoring_opportunities = opportunities if pos == 'RB' else catches
        sim_tds[sim] = np.random.binomial(n=max(1, scoring_opportunities), p=dynamic_prob)

    # Extract milestone probabilities for each player profile
    p_300_pass = np.mean(sim_yards >= 300.0) * 100.0 if pos == 'QB' else 0.0
    p_100_rec  = np.mean(sim_yards >= 100.0) * 100.0 if pos == 'WR' else 0.0
    p_100_rush = np.mean(sim_yards >= 100.0) * 100.0 if pos == 'RB' else 0.0

    if pos == 'QB':
      p_high_volume = np.mean(sim_volume >= 35) * 100.0
    elif pos == 'WR':
      p_high_volume = np.mean(sim_volume >= 10) * 100.0
    else:
      p_high_volume = np.mean(sim_volume >= 20) * 100.0

    player_profiles.append({
      'Player_Name': player['player'],
      'Team': player['team'].upper(),
      'Position': pos,
      'Opponent': player['opponent'].upper() if pd.notna(player['opponent']) else 'UNKNOWN',
      'Pass_DVOA_Matchup': round(pass_dvoa, 3),
      'Rush_DVOA_Matchup': round(rush_dvoa, 3),
      'Expected_Touchdowns': round(np.mean(sim_tds), 2),
      'High_Volume_Probability_%': round(p_high_volume, 1),
      'Prob_Over_300_Pass_Yds_%': round(p_300_pass, 1),
      'Prob_Over_100_Rec_Yds_%': round(p_100_rec, 1),
      'Prob_Over_100_Rush_Yds_%': round(p_100_rush, 1)
    })

  return pd.DataFrame(player_profiles)