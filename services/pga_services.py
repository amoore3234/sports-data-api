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
  dk_data = pd.read_csv('pga_data/dk_salaries.csv')
  df_salary = pd.DataFrame(dk_data)
  df = pd.DataFrame(data)

  target_rating = 76.4
  target_slope = 144

  round_one_score = df[df['Rounds'] == 1]['Scores']
  round_two_score = df[df['Rounds'] == 2]['Scores']
  round_one_scrambling = df[df['Rounds'] == 1]['Scrambling']
  round_two_scrambling = df[df['Rounds'] == 2]['Scrambling']

  base_patential_round_one = target_rating - round_one_score - round_one_scrambling * 4.5
  base_patential_round_two = target_rating - round_two_score - round_two_scrambling * 4.5

  df['Considering_Slope_Round_One'] = target_rating + ((base_patential_round_one - target_rating) * (target_slope / 113))
  df['Considering_Slope_Round_Two'] = target_rating + ((base_patential_round_two - target_rating) * (target_slope / 113))
  df_clean_one = df.dropna(subset=['Considering_Slope_Round_One'])
  df_clean_two = df.dropna(subset=['Considering_Slope_Round_Two'])
  df_list = [df_clean_one, df_clean_two]
  df_combined = pd.merge(df_clean_one, df_clean_two, on='Player', how='left')

  lists = []
  for index_one, index_two in zip(df_clean_one.index, df_clean_two.index):
    lists.append(df.at[index_one, 'Considering_Slope_Round_One'] + df.at[index_two, 'Considering_Slope_Round_Two'])

  df_combined['Total_Score'] = lists
  print(df_combined[['Player', 'Total_Score']])
  dynamic_cut = df_combined['Total_Score'].nsmallest(70).max()
  df_combined['Made_Cut'] = df_combined['Total_Score'] <= dynamic_cut

  salary_lookup = df_salary.set_index('Player')['Salary'].to_dict()
  df_combined['Salary'] = df_combined['Player'].map(salary_lookup)

  is_nan = df_combined['Salary'].isna()
  num_nans = is_nan.sum()

  random_values = np.random.randint(7000, 10001, size=num_nans)
  df_combined.loc[is_nan, 'Salary'] = random_values
  print(df_combined[['Player', 'Salary', 'Made_Cut', 'Total_Score']])

  lineup_count = 0
  lineup_list = []
  lineup = {}

  while lineup_count < 3:
    available_indices = list(df_combined.index)
    player_key = 1
    player_salary = 0
    while player_key <= 6:
      idx = np.random.choice(available_indices)
      lineup[player_key] = df_combined.loc[idx, 'Player']
      available_indices.remove(idx)
      player_salary += df_combined.loc[idx, 'Salary']
      player_key += 1
    if player_salary <= 50000:
      lineup_list.append(lineup)
      lineup_count += 1
  print(f"Lineup selection: {lineup_list}")

  # df['Tournament_Potential'] = df.apply(calculate_slope_potential, rating=target_rating, slope=target_slope, axis=1)

  # print(f"Round 1 Scores:\n{round_one_score}\n")
  # print(f"Round 2 Scores:\n{round_two_score}\n")

  # df.to_csv('pga_data/pebble_beach_potential.csv', index=False)
  print(df[['Player', 'Scores', 'Tournament_Potential']])


def calculate_slope_potential(row, rating, slope):

  base_potential = rating - row['Scores'] - row['Scrambling'] * 4.5

  slope_adjustment = slope / 113

  adjusted_score = rating + ((base_potential - rating) * slope_adjustment)

  # if round_one_score:
  #   rounds_total_score[ += adjusted_score
  # if round_two_score:
  #   rounds_total_score += adjusted_score

  # if rounds_total_score < 142:
  #   print(f"{row['Player']} has a strong potential to win with a score of {rounds_total_score}.")

  return round(adjusted_score, 1)

def calculate_potential(row, rating):
  mean_sg = row['mean']
  std_sg = row['std']

  expected_score = rating - mean_sg

  ceiling = expected_score - (2 * std_sg)
  floor = expected_score + (2 * std_sg)

  return pd.Series([round(expected_score, 1), round(ceiling, 1), round(floor, 1)])

