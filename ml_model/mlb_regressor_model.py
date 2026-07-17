import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import services.mlb_service as mlb_service
from util.mlb_stats_data_util import calculate_on_field_hitter_stats, calculate_on_field_pitcher_stats, fetch_mlb_live_stats

def get_live_mlb_predictions(lineup_df, filter_name, target_metric='strikeouts') -> pd.DataFrame:
  """
  Predicts raw on-field performance metrics without tracking fantasy data.

  - Pitcher: 'strikeouts' (Continuous)
  - Hitter: 'home_run_prob' (Binary Probability) or 'runs' (Continuous)

  Returns:
    DataFrame: A DataFrame containing the predicted performance metrics.
  """
  if len(lineup_df) < 50:
        print(f"WARNING: Since the dataset only has {len(lineup_df)} rows, "
              "at least 50 records are required to achieve a positive R-squared score.")
  # 1. Load your historical performance dataset over the previous 6 days
  tonights_slate_date = "2026-07-22"

  # Fetches actual athletic box score counts instead of DFS points
  live_game_data = fetch_mlb_live_stats(tonights_slate_date)
  player_stats_lookup = live_game_data.set_index('name').to_dict()

  # =====================================================================
  # 🎯 STEP 2: POSITION & TARGET AWARE FEATURE SPACE SELECTION
  # =====================================================================
  if filter_name == 'pitcher':
      features = [
          'pitcher_k_9',
          'pitcher_era',
          'projected_matchup_strikeouts',
          'pitcher_dominance_index',
          'opposing_woba',
          'opposing_xwoba',
          'opposing_team_implied_total'
      ]
      lineup_df['actual_game_strikeouts'] = lineup_df['name'].str.strip().map(player_stats_lookup['current_strikeouts'])

      # Target raw strikeout volume directly
      target_column = 'actual_game_strikeouts'
      lineup_df = calculate_on_field_pitcher_stats(lineup_df)
    #   lineup_df = mlb_service.process_pitcher_data_for_model(lineup_df)
  else:
      features = [
          'sim_scaled_volume_iso',
          'sim_scaled_volume_xwoba',
          'sim_scaled_barrel_leverage',
          'relative_player_volume_delta',
          'simulated_stack_leverage_index',
          'platoon_factor',
          'air_clash_factor',
          'pitcher_weakness_ceiling_multiplier',
          'final_weighted_projection_signal'
      ]
      # Hitter Branch: Switch target column to match on-field stats
      if target_metric == 'home_run_prob':
          target_column = 'actual_game_home_runs'
      else:
          target_column = 'actual_game_runs_scored'

      lineup_df['actual_game_home_runs'] = lineup_df['name'].str.strip().map(player_stats_lookup['current_home_runs'])
      lineup_df['actual_game_runs_scored'] = lineup_df['name'].str.strip().map(player_stats_lookup['current_runs'])
      lineup_df = calculate_on_field_hitter_stats(lineup_df)
      lineup_df = mlb_service.process_batter_data_model(lineup_df)

  if not live_game_data.empty:
      print(f"\n--- Top 10 Leaders in {target_metric.upper()} Over The Last 6 Games ---")
      if target_metric == 'home_run_prob':
          live_game_data = live_game_data.groupby(['player_id', 'name', 'team']).agg({
              'current_home_runs': 'sum'
          }).reset_index()
          live_game_data = live_game_data[['player_id', 'name', 'team', 'current_home_runs']]
          print(live_game_data.sort_values(by='current_home_runs', ascending=False).head(10))
      elif target_metric == 'strikeouts':
        # Group by the unique player ID and name to merge multi-day game logs
        live_game_data = live_game_data.groupby(['player_id', 'name', 'team']).agg({
            'current_strikeouts': 'sum'
        }).reset_index()
        live_game_data = live_game_data[['player_id', 'name', 'team', 'current_strikeouts']]
        print(live_game_data.sort_values(by='current_strikeouts', ascending=False).head(10))
      elif target_metric == 'runs_prob':
        live_game_data = live_game_data.groupby(['player_id', 'name', 'team']).agg({
            'current_runs': 'sum'
        }).reset_index()
        live_game_data = live_game_data[['player_id', 'name', 'team', 'current_runs']]
        print(live_game_data.sort_values(by='current_runs', ascending=False).head(10))

  # Safety cleaning layer: Strip out empty or null rows
  lineup_df = lineup_df.dropna(subset=features + [target_column])

  X = lineup_df[features].copy()
  y = pd.to_numeric(lineup_df[target_column], errors='coerce').fillna(0.0)

  # Clean numeric scale restorations for pitchers
  if filter_name == 'pitcher':
      for col in ['pitcher_dominance_index', 'projected_matchup_strikeouts', 'pitcher_k_9']:
          if col in X.columns:
            # If the column has highly compressed values, normalize via log transformations
            if X[col].max() < 0.1:
                X[col] = np.log1p(X[col] * 1000)

      X['vegas_scaled_dominance'] = X['pitcher_dominance_index'] / (X['opposing_team_implied_total'] + 0.1)
      if 'vegas_scaled_dominance' not in features:
          features.append('vegas_scaled_dominance')

  is_classification = target_metric in ['home_run_prob', 'runs_prob']

  if is_classification:
      # Converts continuous counting statistics (e.g. 2 runs) into binary 1 or 0
      y = np.where(y > 0.0, 1, 0)

  # Split data pools cleanly
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  X_all_scaled = scaler.transform(X)

  # =====================================================================
  # 🎯 STEP 4: MODEL ALGORITHM MATRIX SELECTION SWITCH
  # =====================================================================
  if is_classification:
    # Build Classifier to pull explicit percentage arrays
    rf_model = RandomForestClassifier(
        n_estimators=600, max_depth=5, max_features=0.25, min_samples_leaf=8, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train.astype(int))

    y_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
    lineup_df['projected_stat_metric'] = rf_model.predict_proba(X_all_scaled)[:, 1]

    print(f"\nModel Evaluation (HITTER {target_metric.upper()} Probability Accuracy):")
    print(f"Brier Score Loss: {brier_score_loss(y_test, y_pred):.4f}")
    print(f"Log Loss Density: {log_loss(y_test, y_pred):.4f}")
  else:
      # Strikeouts and Runs Scored use continuous Regressors
      rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,                  # Shallower trees provide better generalization on small slates
        max_features='sqrt',          # Limits feature choices per split to force tree diversity
        min_samples_leaf=2,           # Smooths out extreme node predictions
        criterion='squared_error',
        random_state=42,
        n_jobs=-1
      )

      # Apply standard log transform to hitter runs to stabilize skewed scoring distributions
      rf_model.fit(X_train_scaled, y_train)

      # Process output predictions with inversion layers where applicable
      if filter_name == 'pitcher':
          y_pred = np.maximum(rf_model.predict(X_test_scaled), 0.0)
          lineup_df['projected_stat_metric'] = np.maximum(rf_model.predict(X_all_scaled), 0.0)
      else:
          y_pred = np.expm1(np.maximum(rf_model.predict(X_test_scaled), 0.0))
          lineup_df['projected_stat_metric'] = np.expm1(np.maximum(rf_model.predict(X_all_scaled), 0.0))

      print(f"\nModel Evaluation ({filter_name.upper()} {target_metric.upper()} Counting Accuracy):")
      print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
      print(f"R-squared Efficiency Score: {r2_score(y_test, y_pred):.2f}")

  # Extract feature importance rankings safely
  importances = rf_model.feature_importances_
  feature_importance_list = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

  print(f"\n--- Sorted {target_metric.upper()} Model Feature Importances ---")
  for feature, importance in feature_importance_list[:5]:
      print(f"Feature: {feature:35} | Importance: {importance:.4f}")

  # Rename final column output cleanly to match your on-field metrics
  lineup_df.rename(columns={'projected_stat_metric': f'projected_{target_metric}'}, inplace=True)

  # 6. Extract tree structures cleanly for standard deviation/variance modeling
  if is_classification:
      # Classifiers MUST evaluate variance using tree probability vectors (predict_proba)
      # This isolates individual tree probability votes for the positive case ([:, 1])
      all_tree_probs = np.array([tree.predict_proba(X_all_scaled)[:, 1] for tree in rf_model.estimators_])

      # Calculate standard deviation across the probability arrays
      # This guarantees every output value sits safely in a valid [0.0 to 0.50] boundary range
      lineup_df[f'projected_{target_metric}_std'] = np.std(all_tree_probs, axis=0)
  else:
      # Regressors evaluate variance using continuous prediction metrics
      all_tree_preds = np.array([tree.predict(X_all_scaled) for tree in rf_model.estimators_])

      # Apply exponential inversions ONLY to continuous offensive counts (like raw runs or hits)
      if filter_name != 'pitcher' and 'prob' not in target_metric:
          all_tree_preds = np.expm1(np.maximum(all_tree_preds, 0.0))

      lineup_df[f'projected_{target_metric}_std'] = np.std(all_tree_preds, axis=0)

  return lineup_df[['name', 'team', 'position', f'projected_{target_metric}', f'projected_{target_metric}_std']]