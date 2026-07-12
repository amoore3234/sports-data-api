import pandas as pd
import numpy as np
from util.pga_stats_util import get_player_recent_sg_statistics
from util.pga_stats_util import get_seasonal_pga_data
from ml_model.pga_regressor_model import get_pga_scoring_probabilities
from collections import defaultdict

def predict_pga_top_performers() -> pd.DataFrame:
  """Predicts a players skill prior to a PGA tournament.

    Returns:
      DataFrame: Player skill data.
  """

  par_5_birdie_data = pd.read_csv('pga_data/par_5_birdie_stats.csv', encoding='cp1252')
  par_5_birdie_df = pd.DataFrame(par_5_birdie_data)
  par_5_birdie_df['Player'] = par_5_birdie_df['Player'].str.lower()

  tournaments = get_seasonal_pga_data(is_seasonal=True)
  recent_tournaments = get_seasonal_pga_data(is_seasonal=False)
  recent_tournament_df = pd.concat(recent_tournaments, ignore_index=True)
  tournament_df = pd.concat(tournaments, ignore_index=True)

  players_sg_averages_df = get_player_recent_sg_statistics(tournament_df, is_seasonal=True)

  player_round_lookup = defaultdict(dict)
  par_5_birdie_lookup = par_5_birdie_df.set_index('Player')['Percentage']
  for row in players_sg_averages_df.itertuples(index=False):
    # Create an object (dictionary) containing just the stats for this round
    stats_object = {
      "sg_putting": row.seasonal_round_sg_putting_average,
      "sg_around_green": row.seasonal_round_sg_around_green_average,
      "sg_approach": row.seasonal_round_sg_approach_average,
      "sg_off_the_tee": row.seasonal_round_sg_off_the_tee_average,
      "sg_tee_to_green": row.seasonal_round_sg_tee_to_green_average,
      "sg_total": row.seasonal_round_sg_total_average,
      "scrambling": row.seasonal_round_scrambling_average
    }

    player_round_lookup[row.player][row.Rounds] = stats_object

  players_recent_sg_df = get_player_recent_sg_statistics(recent_tournament_df, is_seasonal=False)

  players_recent_sg_df['par_5_birdie_average'] = players_recent_sg_df['player'].map(par_5_birdie_lookup)
  players_recent_sg_df['sg_seasonal_putting_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_putting'),
    axis=1
  )
  players_recent_sg_df['sg_seasonal_approach_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_approach'),
    axis=1
  )
  players_recent_sg_df['sg_seasonal_around_green_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_around_green'),
    axis=1
  )
  players_recent_sg_df['sg_seasonal_off_the_tee_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_off_the_tee'),
    axis=1
  )
  players_recent_sg_df['sg_seasonal_tee_to_green_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_tee_to_green'),
    axis=1
  )
  players_recent_sg_df['sg_seasonal_total_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('sg_total'),
    axis=1
  )
  players_recent_sg_df['seasonal_scrambling_average'] = players_recent_sg_df.apply(
    lambda row: player_round_lookup.get(row['player'], {}).get(row['Rounds'], {}).get('scrambling'),
    axis=1
  )

  players_recent_sg_df = get_pga_scoring_probabilities(players_recent_sg_df)

  aggregation_rules = {
      'course_par': 'first',
      'Eagle_%': 'mean',
      'Birdie_%': 'mean',
      'Par_%': 'mean',
      'Bogey_%': 'mean',
      'Double_%': 'mean'
  }
  # Group by Player to collapse the 4 rows into 1 single unique row per golfer
  players_recent_sg_df = players_recent_sg_df.groupby('player', as_index=False).agg(aggregation_rules)
  players_recent_sg_df = run_pga_monte_carlo_tournament_simulation(players_recent_sg_df)
  return players_recent_sg_df

def simulate_rounds(player_row, num_rounds, num_sims) -> np.ndarray:
  """Simulates a specific number of rounds (18 holes each) for a golfer using DataFrame columns.

    Parameters:
      player_row: Plahyer data.
      num_rounds: The number of rounds in a tournament.
      num_sims: Number of simulation rounds.

    Returns:
      NDArray: 2D Matrix of hole performances.
  """
  outcomes = ['eagle', 'birdie', 'par', 'bogey', 'double_plus']
  total_holes = num_rounds * 18

  raw_p = np.array([
    player_row['Eagle_%'],
    player_row['Birdie_%'],
    player_row['Par_%'],
    player_row['Bogey_%'],
    player_row['Double_%']
  ], dtype=np.float64)

  clipped_p = np.clip(raw_p, 0.0001, None)

  # Re-normalize the matrix so the values cleanly sum up to exactly 1.0
  normalized_p = clipped_p / np.sum(clipped_p)

  # Pull out individual column metrics directly from the row Series
  sim_holes = np.random.choice(
      outcomes,
      size=(num_sims, total_holes),
      p=normalized_p
  )

  return sim_holes

def calculate_round_scores(sim_holes) -> list[int]:
  """Translates raw hole outcomes into cumulative strokes relative to par.

    Parameters:
      sim_holes: Hole performances simulated in a tournament.

    Returns:
      list[int]: A list of integers (strokes).
  """
  stroke_map = {'eagle': -2, 'birdie': -1, 'par': 0, 'bogey': 1, 'double_plus': 2}
  vectorized_strokes = np.vectorize(lambda x: stroke_map.get(x, 0))
  return np.sum(vectorized_strokes(sim_holes), axis=1)

def run_pga_monte_carlo_tournament_simulation(golfers_df, num_sims=1000) -> pd.DataFrame:
  """Runs a tournament simulation directly utilizing a pandas DataFrame.

    Parameters:
      golfers_df: DataFrame containing player metrics.
      num_sims: Number of iterations to simulate a live game.

    Returns:
      DataFrame: Simulation results
  """
  num_players = len(golfers_df)

  # Simulate the first 2 rounds (36 holes) for the entire field
  r12_strokes = np.zeros((num_sims, num_players))
  sim_birdie_counts = np.zeros((num_sims, num_players))

  for idx in range(num_players):
    player_row = golfers_df.iloc[idx]
    simulated_holes = simulate_rounds(player_row, num_rounds=2, num_sims=num_sims)
    r12_strokes[:, idx] = calculate_round_scores(simulated_holes)

  # Determine the Cut Line for every individual simulation
  made_cut_matrix = np.zeros((num_sims, num_players), dtype=bool)

  for sim in range(num_sims):
    sim_scores = r12_strokes[sim, :]
    sim_birdie_counts[sim, idx] += np.sum(r12_strokes[idx, :] == -1)
    cut_threshold_score = np.partition(sim_scores, 64)[64] if num_players > 64 else np.max(sim_scores)
    made_cut_matrix[sim, :] = sim_scores <= cut_threshold_score

  # Simulate the Weekend (Rounds 3 and 4)
  total_tournament_strokes = np.zeros((num_sims, num_players))
  cut_made_counts = np.zeros(num_players)

  for idx in range(num_players):
    player_row = golfers_df.iloc[idx]
    weekend_holes = simulate_rounds(player_row, num_rounds=2, num_sims=num_sims)

    weekend_strokes = calculate_round_scores(weekend_holes)

    for sim in range(num_sims):
      if made_cut_matrix[sim, idx]:
        total_tournament_strokes[sim, idx] = r12_strokes[sim, idx] + weekend_strokes[sim]
        sim_birdie_counts[sim, idx] += np.sum(weekend_strokes == -1)
        cut_made_counts[idx] += 1
      else:
        total_tournament_strokes[sim, idx] = 999

  top_10_counts = np.zeros(num_players)
  top_20_counts = np.zeros(num_players)

  for sim in range(num_sims):
    sim_leaderboard = total_tournament_strokes[sim, :]

    # Find the threshold scores for 10th and 20th place (accounting for ties)
    top_10_threshold = np.partition(sim_leaderboard, 9)[9]
    top_20_threshold = np.partition(sim_leaderboard, 19)[19]

    # Flag players who beat or match those threshold scores
    top_10_counts += (sim_leaderboard <= top_10_threshold) & (sim_leaderboard < 99)
    top_20_counts += (sim_leaderboard <= top_20_threshold) & (sim_leaderboard < 99)

  # Compile DataFrame with exact GPP metrics
  results = []
  for idx in range(num_players):
    player_row = golfers_df.iloc[idx]
    player_name = player_row['player']

    valid_final_scores = total_tournament_strokes[:, idx][total_tournament_strokes[:, idx] != 999]
    avg_final_score = np.mean(valid_final_scores) if len(valid_final_scores) > 0 else (player_row['course_par'] * 4 + 10)

    results.append({
      'player': player_name,
      'make_cut_%': round((cut_made_counts[idx] / num_sims) * 100, 1),
      'avg_birdies_made': round(np.mean(sim_birdie_counts[:, idx]), 1),
      'expected_final_strokes': round(avg_final_score, 1),
      'top_10_finish_%': round((top_10_counts[idx] / num_sims) * 100, 1),
      'top_20_finish_%': round((top_20_counts[idx] / num_sims) * 100, 1)
    })

  results_df = pd.DataFrame(results).sort_values(by='top_10_finish_%', ascending=False)
  print(results_df)
  return results_df