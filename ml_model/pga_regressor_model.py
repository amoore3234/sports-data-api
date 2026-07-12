import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

course_difficulty = {
  'par_5': 3, 'rating': 76.8, 'slope': 151, 'par': 71
}

def get_pga_scoring_probabilities(tournament_statistics_df) -> pd.DataFrame:
  """Generate probability scores for players in a PGA tournament.

    This function uses the Random Forest Regressor model to calculate a players'
    performance based on past performances.

    Parameters:
      tournament_statistics_df: Golf tournament data.

    Returns:
      DataFrame: Includes predictive scoring probabilities.
  """
  # Include additional training data
  tournament_statistics_df = add_difficulty_feature(tournament_statistics_df, course_difficulty)
  tournament_statistics_df = get_par_5_performance(tournament_statistics_df)

  features = [
    'sg_seasonal_putting_average',
    'sg_seasonal_off_the_tee_average',
    'sg_seasonal_tee_to_green_average',
    'sg_seasonal_approach_average',
    'par_5_birdie_average',
    'Round_2_Cut_Pressure',
    'par_5_count',
    'course_difficulty_factor',
    'Rounds'
  ]

  X = tournament_statistics_df[features]
  y_tee_to_green = tournament_statistics_df['current_round_sg_tee_to_green']
  y_putting = tournament_statistics_df['current_round_sg_putting']
  y_approach = tournament_statistics_df['current_round_sg_approach']

  # Split to evaluate accuracy
  X_train, X_test, y_train_ttg, y_test_ttg = train_test_split(X, y_tee_to_green, test_size=0.2, random_state=42)
  _, _, y_train_putt, y_test_putt = train_test_split(X, y_putting, test_size=0.2, random_state=42)
  _, _, y_train_approach, y_test_approach = train_test_split(X, y_approach, test_size=0.2, random_state=42)

  # Instantiate models with constraints to avoid overfitting on single-round variance
  rf_ttg = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
  rf_putting = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
  rf_approach = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)

  rf_ttg.fit(X_train, y_train_ttg)
  rf_putting.fit(X_train, y_train_putt)
  rf_approach.fit(X_train, y_train_approach)

  # Run the trained Regressors to get custom round-level expected baselines
  pred_ttg = rf_ttg.predict(tournament_statistics_df[features])
  pred_putting = rf_putting.predict(tournament_statistics_df[features])
  pred_approach = rf_approach.predict(tournament_statistics_df[features])

  # Build the final baseline matrix
  probability_results = []
  for idx, (i, row) in enumerate(tournament_statistics_df.iterrows()):
    scaled_probs, p5_probs = convert_sg_to_probabilities(
      sg_off_the_tee=(pred_ttg[idx] - pred_approach[idx]),
      sg_approach=pred_approach[idx],
      sg_putting=pred_putting[idx],
      scrambling_pct=row['seasonal_scrambling_average'],
      par_5_birdie_pct=row['par_5_birdie_average'],
      course_difficulty=row['course_difficulty_factor']
    )

    p5_count = int(row['par_5_count'])
    std_count = 18 - p5_count

    # Calculate individual hole weights
    std_weight = std_count / 18.0
    p5_weight = p5_count / 18.0

    # Combine both arrays into one single integrated, math-safe distribution array
    combined_probs = (scaled_probs * std_weight) + (p5_probs * p5_weight)

    probability_results.append({
      'player': row['player'],
      'course_par': row['course_par'],
      'Predicted_Round_SG_TTG': round(pred_ttg[i], 2),
      'Predicted_Round_SG_Putt': round(pred_putting[i], 2),
      'Eagle_%': round(combined_probs[0] * 100, 2),
      'Birdie_%': round(combined_probs[1] * 100, 2),
      'Par_%': round(combined_probs[2] * 100, 2),
      'Bogey_%': round(combined_probs[3] * 100, 2),
      'Double_%': round(combined_probs[4] * 100, 2)
    })

  return pd.DataFrame(probability_results)

def get_par_5_performance(df):
  """Generate Par 5 performance.

    Parameters:
      df: Tournament DataFrame data

    Returns:
      DataFrame: Tournament DataFrame
  """

  df['par_5_count'] = course_difficulty['par_5']
  df['course_par'] = course_difficulty['par']

  # Introduce Round 2 Cut Line Pressure Flag
  # 1 if it's Friday (Round 2) and the player is a fringe golfer (Long-term SG Total below 0.5)
  # 0 otherwise. This allows the model to learn performance decay on cut-day.
  df['Round_2_Cut_Pressure'] = np.where(
      (df['Rounds'] == 2) & (df['sg_seasonal_total_average'] < 0.5),
      1,
      0
  )

  return df

def convert_sg_to_probabilities(
    sg_off_the_tee, sg_approach, sg_putting, scrambling_pct, par_5_birdie_pct, course_difficulty
) -> np.ndarray:
  """Advanced probability mapping that isolates SG:Approach to drive

    High-upside birdie and eagle metrics.

    Parameters:
      sg_off_the_tee: Strokes Gained: Off The Tee metric.
      sg_approach: Strokes Gained: Approach metric.
      sg_putting: Strokes Gained: Putting metric.
      scrambling_pct: Scrambling Percentage metric.
      par_5_birdie_pct: Par 5 Birdie Percentage metric.
      course_difficulty: Course rating and difficulty score.

    Returns:
      NDArray: Array of scoring predictions.
  """
  baseline = {
    "eagle": 0.005,
    "birdie": 0.215,
    "par": 0.630,
    "bogey": 0.135,
    "double": 0.015,
  }

  base_p5  = {'eagle': 0.040, 'birdie': 0.420, 'par': 0.440, 'bogey': 0.090, 'double': 0.010}

  # Scale round-level predictions down to single-hole metrics
  app_effect = sg_approach / 18.0
  off_the_tee_effect = sg_off_the_tee / 18.0
  putting_effect = sg_putting / 18.0

  # SG:Approach heavily drives Birdie and Eagle creation
  p_eagle = max(0.001, baseline["eagle"] + (app_effect * 0.06))
  p_birdie = max(
    0.05,
    baseline["birdie"]
    + (app_effect * 0.55)  # Highest weight given to iron play
    + (off_the_tee_effect * 0.45)
    + (putting_effect * 0.35),
  )

  # Scrambling and Off-the-Tee skill keep bogeys off the card
  scrambling_bonus = (scrambling_pct - 0.58) * 0.10
  p_bogey = max(
    0.01,
    baseline["bogey"]
    - (app_effect * 0.25)
    - (off_the_tee_effect * 0.45)
    - (putting_effect * 0.30)
    - scrambling_bonus,
  )
  p_double = max(
      0.001,
      baseline["double"] - (off_the_tee_effect * 0.15) - (scrambling_bonus * 0.2),
  )

  # Convert course difficulty multiplier to an exponential decay function
  # to guarantee probabilities never drop below zero or exceed limits.
  # If difficulty is 3.398, np.exp(1 - 3.398) = 0.09 (birdies compress naturally but stay positive)
  difficulty_decay = np.exp(1.0 - course_difficulty)

  p_eagle *= difficulty_decay
  p_birdie *= difficulty_decay
  p_double *= course_difficulty

  # Calculate par baseline
  p_par = 1.0 - (p_eagle + p_birdie + p_bogey + p_double)

  # Force a strict floor of 0.01% so no parameter ever turns negative
  p_eagle = max(0.0001, p_eagle)
  p_birdie = max(0.001, p_birdie)
  p_par    = max(0.01, p_par)
  p_double = max(0.0001, p_double)

  # Matrix normalization to guarantee exactly 1.0 sum
  raw_probs = np.array([p_eagle, p_birdie, p_par, p_bogey, p_double], dtype=np.float64)
  scaled_probs = raw_probs / np.sum(raw_probs)

  p5_birdie_bonus = (par_5_birdie_pct - 0.45) * 0.50
  p5_birdie = max(0.05, base_p5['birdie'] + (app_effect * 0.50) + (putting_effect * 0.50) + p5_birdie_bonus)
  p5_eagle = max(0.001, base_p5['eagle'] + (app_effect * 0.30) + (p5_birdie_bonus * 0.10))
  p5_bogey = max(0.01, base_p5['bogey'] - (app_effect * 0.20))
  p5_par = max(0.01, 1.0 - (p5_eagle + p5_birdie + p5_bogey + 0.010))

  p5_probs = np.array([p5_eagle, p5_birdie, p5_par, p5_bogey, 0.010])
  p5_probs /= np.sum(p5_probs) # Normalize

  return scaled_probs, p5_probs

def add_difficulty_feature(df, difficulty):

  # Execute your mathematical formula function
  difficulty_mapping = calculate_course_difficulty(
    course_rating=difficulty['rating'],
    slope_rating=difficulty['slope'],
    course_par=difficulty['par']
  )

  # Map the unique float numbers directly onto the DataFrame rows
  df['course_difficulty_factor'] = difficulty_mapping
  return df

def calculate_course_difficulty(course_rating, slope_rating, course_par=72) -> float:
  """Translates USGA Course Rating and Slope into a standardized scoring delta

    relative to a neutral PGA Tour baseline (+1.5 strokes over par).

    Parameters:
      course_rating: Average score relative to Par.
      slope_rating: Course's difficulty score.
      course_par: Course's Par.

    Returns:
      Float: Course difficulty value.
  """
  # Calculate the baseline rating difference from par
  rating_delta = course_rating - course_par

  # Factor in the slope difficulty scaling factor relative to standard (113)
  slope_scalar = slope_rating / 130.0

  raw_difficulty = (rating_delta * 0.25) + (slope_scalar * 0.5)

  # Anchor around 1.0
  standardized_factor = 1.0 + (raw_difficulty - 0.5) * 0.1

  return round(standardized_factor, 3)