import pandas as pd
import numpy as np
import scipy.stats as stats

def player_expected_score():

  data = {
  'Player': ['Collin Morikawa', 'Collin Morikawa', 'Shane Lowry', 'Shane Lowry'],
  'SG_Total': [7.938, 2.788, 1.25, 0.526]
  }
  df = pd.DataFrame(data)

  player_stats = df.groupby('Player')['SG_Total'].agg(['mean', 'std']).reset_index()

  course_rating = 73.4

  player_stats[['Expected_Score', 'Ceiling', 'Floor']] = player_stats.apply(calculate_potential, rating=course_rating, axis=1)

  print(player_stats[['Player', 'Expected_Score', 'Ceiling', 'Floor']])

def player_expected_score_with_course_difficulty():
  data = pd.read_csv('pga_data/pebble_beach_stats.csv')
  df = pd.DataFrame(data)

  target_rating = 76.4
  target_slope = 144

  df['Tournament_Potential'] = df.apply(calculate_slope_potential, rating=target_rating, slope=target_slope, axis=1)
  df.to_csv('pga_data/pebble_beach_potential.csv', index=False)
  print(df[['Player', 'SG Total', 'Tournament_Potential']])


def calculate_slope_potential(row, rating, slope):
  base_potential = rating - row['SG Total'] - row['Scrambling'] * 4.5

  slope_adjustment = slope / 113

  adjusted_score = rating + ((base_potential - rating) * slope_adjustment)

  return round(adjusted_score, 1)

def calculate_potential(row, rating):
  mean_sg = row['mean']
  std_sg = row['std']

  expected_score = rating - mean_sg

  ceiling = expected_score - (2 * std_sg)
  floor = expected_score + (2 * std_sg)

  return pd.Series([round(expected_score, 1), round(ceiling, 1), round(floor, 1)])

