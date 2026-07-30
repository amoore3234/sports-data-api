import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def get_nfl_player_baselines(historical_game_logs_df) -> pd.DataFrame:
  """
  Trains independent, position-isolated regressors to prevent cross-contamination
  and align metric scales before the simulation runs.

  Args:
    historical_game_logs_df (pd.DataFrame): A dataframe containing historical game logs for NFL players.

  Returns:
    pd.DataFrame: The input dataframe augmented with predicted volume, yards, touchdowns, and volume
  """
  features = [
    'avg_target_share',
    'avg_air_yards_share',
    'avg_receiving_epa',
    'avg_carries',
    'avg_rushing_yards',
    'avg_rushing_epa',
    'avg_passing_yards',
    'avg_passing_tds',
    'avg_passing_epa',
    'opp_rush_dvoa',
    'opp_pass_dvoa',
    'implied_team_total',
    'seasonal_rz_targets',
    'seasonal_rz_carries',
    'team_redzone_success_rate'
  ]

  # Pre-assign clean, permanent position labels based on historical baselines
  historical_game_logs_df['position_assigned'] = np.where(historical_game_logs_df['avg_passing_yards'] > 50.0, 'QB',
                                                  np.where(historical_game_logs_df['avg_carries'] > 4.0, 'RB', 'WR'))

  # Initialize empty prediction destination columns
  for col in ['pred_vol', 'pred_yds', 'pred_tds', 'pred_vol_std']:
    historical_game_logs_df[col] = 0.0

  # Isolate positions into distinct sub-frames for independent training pipelines
  positions_pool = ['QB', 'RB', 'WR']

  # --- START POSITION-STRATIFIED LOOP ---
  for pos in positions_pool:
    df_pos_sub = historical_game_logs_df[historical_game_logs_df['position_assigned'] == pos].copy()
    if df_pos_sub.empty:
      continue

    X_sub = df_pos_sub[features].fillna(0.0)

    # Target assignment adjustments tailored strictly to the current active position
    if pos == 'QB':
      y_vol = df_pos_sub['recent_pass_attempts']
      y_yds = df_pos_sub['avg_passing_yards']
      y_tds = df_pos_sub['avg_passing_tds']
    elif pos == 'RB':
      y_vol = df_pos_sub['avg_carries']
      y_yds = df_pos_sub['avg_rushing_yards']
      y_tds = df_pos_sub['historical_td_prob_per_opportunity']
    else: # Wide Receivers & Tight Ends
      y_vol = df_pos_sub['recent_targets']
      y_yds = df_pos_sub['avg_receiving_yards']
      y_tds = df_pos_sub['actual_game_total_tds']

    # Instantiate highly localized, targeted model architectures
    rf_vol = RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_yds = RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_tds = RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1)

    # Fit models strictly on the filtered position rows
    rf_vol.fit(X_sub, y_vol)
    rf_yds.fit(X_sub, y_yds)
    rf_tds.fit(X_sub, y_tds)

    # Map predictions back to the master dataframe matching the specific position mask
    mask = historical_game_logs_df['position_assigned'] == pos
    historical_game_logs_df.loc[mask, 'pred_vol'] = rf_vol.predict(X_sub)
    historical_game_logs_df.loc[mask, 'pred_yds'] = rf_yds.predict(X_sub)
    historical_game_logs_df.loc[mask, 'pred_tds'] = rf_tds.predict(X_sub)

    # Extract precise tree standard deviations across the localized sub-models
    all_tree_vol_preds = np.array([tree.predict(X_sub) for tree in rf_vol.estimators_])
    historical_game_logs_df.loc[mask, 'pred_vol_std'] = np.std(all_tree_vol_preds, axis=0)

  return historical_game_logs_df