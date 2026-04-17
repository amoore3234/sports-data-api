import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

def player_expected_score_at_course():
  """Calculates a player's projected scores per round at a specific PGA course.

  Returns:
      csv: A report of projected scores per round for each player.
  """
  # Prepare the data
  pebble_data = pd.read_csv('pga_data/pebble_beach_stats.csv')
  phoenix_open_data = pd.read_csv('pga_data/wm_phoenix_open_stats.csv')
  genesis_invitational_data = pd.read_csv('pga_data/genesis_invitational_stats.csv')
  cognizant_class_data = pd.read_csv('pga_data/cognizant_classic_stats.csv')
  arnold_palmer_data = pd.read_csv('pga_data/arnold_palmer_stats.csv')
  players_championships_data = pd.read_csv('pga_data/players_championships_stats.csv')
  valspar_championship_data = pd.read_csv('pga_data/valspar_championship_stats.csv')
  texas_childrens_houston_open_data = pd.read_csv('pga_data/texas_childrens_houston_open_stats.csv')
  valero_texas_open_data = pd.read_csv('pga_data/valero_texas_open_stats.csv')
  dk_data = pd.read_csv('pga_data/dk_salaries.csv')
  fd_data = pd.read_csv('pga_data/fd_salaries.csv')
  df_salary = pd.DataFrame(dk_data)
  df_salary['Name'] = df_salary['Name'].str.lower()
  fd_salary_df = pd.DataFrame(fd_data)
  phoenix_df = pd.DataFrame(phoenix_open_data)
  genesis_invitational_df = pd.DataFrame(genesis_invitational_data)
  pebble_beach_df = pd.DataFrame(pebble_data)
  cognizant_classic_df = pd.DataFrame(cognizant_class_data)
  arnold_palmer_df = pd.DataFrame(arnold_palmer_data)
  players_championships_df = pd.DataFrame(players_championships_data)
  valspar_championship_df = pd.DataFrame(valspar_championship_data)
  texas_childrens_houston_open_df = pd.DataFrame(texas_childrens_houston_open_data)
  valero_texas_open_df = pd.DataFrame(valero_texas_open_data)
  tournaments = [phoenix_df, genesis_invitational_df, cognizant_classic_df, pebble_beach_df,
                 arnold_palmer_df, players_championships_df, valspar_championship_df,
                 texas_childrens_houston_open_df, valero_texas_open_df]
  tournament_df = pd.concat(tournaments, ignore_index=True)

  fanduel_lineup = True
  # A course's scoring average and difficulty.
  target_rating = 74.2
  target_slope = 128

  # Calculate a player's current average Strokes Gained and Scrambling statistics per round.
  round_one_total_sg = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Total'].transform('mean')
  round_one_scrambling = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['Scrambling'].transform('mean')
  round_two_total_sg = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Total'].transform('mean')
  round_two_scrambling = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['Scrambling'].transform('mean')
  round_three_total_sg = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Total'].transform('mean')
  round_three_scrambling = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['Scrambling'].transform('mean')
  round_four_total_sg = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Total'].transform('mean')
  round_four_scrambling = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['Scrambling'].transform('mean')

  # Calcaluate a player's base scoring performance per round.
  base_patential_round_one = target_rating - round_one_total_sg - round_one_scrambling * 4.5
  base_patential_round_two = target_rating - round_two_total_sg - round_two_scrambling * 4.5
  base_patential_round_three = target_rating - round_three_total_sg - round_three_scrambling * 4.5
  base_patential_round_four = target_rating - round_four_total_sg - round_four_scrambling * 4.5

  # Calculate a player's projected scoring results per round based on course difficulty
  round_one_result = target_rating + ((base_patential_round_one - target_rating) * (target_slope / 113))
  round_two_result = target_rating + ((base_patential_round_two - target_rating) * (target_slope / 113))
  round_three_result = target_rating + ((base_patential_round_three - target_rating) * (target_slope / 113))
  round_four_result = target_rating + ((base_patential_round_four - target_rating) * (target_slope / 113))

  # Create columns that contain the projected scoring results per round for each player.
  tournament_df['Projected_Round_One_Result'] = round(round_one_result, 0)
  round_one_drop = tournament_df.dropna(subset=['Projected_Round_One_Result'])
  round_one_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_one_results = list(round_one_drop['Projected_Round_One_Result'])

  tournament_df['Projected_Round_Two_Result'] = round(round_two_result, 0)
  round_two_drop = tournament_df.dropna(subset=['Projected_Round_Two_Result'])
  round_two_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_two_results = list(round_two_drop['Projected_Round_Two_Result'])

  tournament_df['Projected_Round_Three_Result'] = round(round_three_result, 0)
  round_three_drop = tournament_df.dropna(subset=['Projected_Round_Three_Result'])
  round_three_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_three_results = list(round_three_drop['Projected_Round_Three_Result'])

  tournament_df['Projected_Round_Four_Result'] = round(round_four_result, 0)
  round_four_drop = tournament_df.dropna(subset=['Projected_Round_Four_Result'])
  round_four_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_four_results = list(round_four_drop['Projected_Round_Four_Result'])

  # Return the final projected scoring results for the first two rounds and the final two rounds.
  first_two_rounds_results = [round(i + j, 1) for i, j in zip(round_one_results, round_two_results)]
  final_two_rounds_results = [round(x + y, 1) for x, y in zip(round_three_results, round_four_results)]

  # Create a list that does not contain duplicate player entries.
  clean_list = list(dict.fromkeys(list(tournament_df['Player'])))

  # Create a new Data Frame that contains projected scores for each player.
  new_tournament_df = tournament_df.iloc[:len(clean_list)].copy()
  new_tournament_df['Player'] = clean_list
  new_tournament_df['Projected_First_Two_Rounds_Result'] = first_two_rounds_results
  new_tournament_df['Projected_Final_Two_Rounds_Result'] = final_two_rounds_results

  # This calculates a dynamic cut line based on the projected scoring results for the first two rounds.
  # It takes the top 70 scores in the series and returns the score needed to make the cut.
  dynamic_cut = new_tournament_df['Projected_First_Two_Rounds_Result'].nsmallest(70).max()

  new_tournament_df['Made_Cut'] = new_tournament_df['Projected_First_Two_Rounds_Result'] <= dynamic_cut

  # This only returns the list of players that are projected to make the cut and their projected final scores.
  mask = new_tournament_df['Made_Cut'] == True
  new_tournament_df.loc[mask, 'Final_Score'] = (
    round(new_tournament_df['Projected_First_Two_Rounds_Result'] + new_tournament_df['Projected_Final_Two_Rounds_Result'], 1)
  )

  # Drop columns that contain null values and columns that are not needed for the final stat sheet.
  remove_nan_df = new_tournament_df.dropna(subset=['Final_Score'])
  lineup_creation_df = remove_nan_df.drop(columns=['SG Total', 'Scrambling', 'Rounds', 'Projected_Round_One_Result',
    'Projected_Round_Two_Result', 'Projected_Round_Three_Result', 'Projected_Round_Four_Result'])

  salary_lookup = df_salary.set_index('Name')['Salary'].to_dict()
  name_id = df_salary.set_index('Name')['Name + ID'].to_dict()

  if fanduel_lineup:
    salary_lookup = fd_salary_df.set_index('Nickname')['Salary'].to_dict()
    name_id = fd_salary_df.set_index('Nickname')['Player ID + Player Name'].to_dict()

  lineup_creation_df['Salary'] = lineup_creation_df['Player'].map(salary_lookup)
  lineup_creation_df['Name_ID'] = lineup_creation_df['Player'].map(name_id)

  lineup_count = 0
  lineup_list = []

  while lineup_count < 30:
    lineup = {}
    available_indices = list(lineup_creation_df.index)
    player_key = 1
    player_salary = 0
    while player_key <= 6:
      idx = np.random.choice(available_indices)
      lineup[player_key] = lineup_creation_df.loc[idx, 'Name_ID']
      player_salary += lineup_creation_df.loc[idx, 'Salary']
      available_indices.remove(idx)
      player_key += 1
    if fanduel_lineup:
      if player_salary <= 60000 and player_salary >= 59000:
        lineup_list.append(lineup)
        lineup_count += 1
    else:
      if player_salary <= 50000 and player_salary >= 49000:
        lineup_list.append(lineup)
        lineup_count += 1
  temp_df = pd.DataFrame({1: [], 2: [], 3: [], 4:[], 5:[], 6:[]})
  lineups_df = pd.DataFrame(lineup_list)
  lineup_df = pd.concat([temp_df, lineups_df], ignore_index=True)

  print(lineup_list)

  lineup_df.to_csv('pga_data/lineups.csv', index=False)
  lineup_creation_df.to_csv('pga_data/tournament_results.csv', index=False)

def calculate_potential(row, rating):
  mean_sg = row['mean']
  std_sg = row['std']

  expected_score = rating - mean_sg

  ceiling = expected_score - (2 * std_sg)
  floor = expected_score + (2 * std_sg)

  return pd.Series([round(expected_score, 1), round(ceiling, 1), round(floor, 1)])

def predict_top_10_performance():
  """Predicts a player's future top 10 performance probabilities.

  The logic trains a Random Forest Classifier model to generate
  probability metrics of a player for predicting a top 10 finish
  for a given tournament.

  Key metrics:
  - Season Strokes Gained Total: A player's average Strokes Gained total per round for the current season.
  - Confusion Matrix: A table used to evaluate the performance of a classification model.
  - Classification Report: A report that includes precision, recall, f1-score, and support.
  - Feature Importance: A metric that indicates the importance of each feature tested within the model.
  """
  pebble_data = pd.read_csv('pga_data/pebble_beach_stats.csv')
  phoenix_open_data = pd.read_csv('pga_data/wm_phoenix_open_stats.csv')
  genesis_invitational_data = pd.read_csv('pga_data/genesis_invitational_stats.csv')
  cognizant_class_data = pd.read_csv('pga_data/cognizant_classic_stats.csv')
  season_average_sg_total = pd.read_csv('pga_data/season_average_sg_total.csv')
  arnold_palmer_data = pd.read_csv('pga_data/arnold_palmer_stats.csv')
  players_championships_data = pd.read_csv('pga_data/players_championships_stats.csv')
  valspar_championship_data = pd.read_csv('pga_data/valspar_championship_stats.csv')
  texas_childrens_houston_open_data = pd.read_csv('pga_data/texas_childrens_houston_open_stats.csv')
  valero_texas_open_data = pd.read_csv('pga_data/valero_texas_open_stats.csv')
  phoenix_df = pd.DataFrame(phoenix_open_data)
  genesis_invitational_df = pd.DataFrame(genesis_invitational_data)
  pebble_beach_df = pd.DataFrame(pebble_data)
  cognizant_classic_df = pd.DataFrame(cognizant_class_data)
  season_average_sg_total_df = pd.DataFrame(season_average_sg_total)
  arnold_palmer_df = pd.DataFrame(arnold_palmer_data)
  players_championships_df = pd.DataFrame(players_championships_data)
  valspar_championship_df = pd.DataFrame(valspar_championship_data)
  texas_childrens_houston_open_df = pd.DataFrame(texas_childrens_houston_open_data)
  valero_texas_open_df = pd.DataFrame(valero_texas_open_data)
  tournaments = [phoenix_df, genesis_invitational_df, cognizant_classic_df, pebble_beach_df,
                 arnold_palmer_df, players_championships_df, valspar_championship_df,
                 texas_childrens_houston_open_df, valero_texas_open_df]
  tournament_df = pd.concat(tournaments, ignore_index=True)

  # Calculate a player's current average Strokes Gained statistics per round.
  round_one_total_sg = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Total'].transform('mean')
  round_two_total_sg = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Total'].transform('mean')
  round_three_total_sg = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Total'].transform('mean')
  round_four_total_sg = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Total'].transform('mean')

  # Create columns that contains the total Strokes Gained per round for each player.
  tournament_df['Round_One_SG_Total'] = round(round_one_total_sg, 1)
  round_one_drop = tournament_df.dropna(subset=['Round_One_SG_Total'])
  round_one_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_one_results = list(round_one_drop['Round_One_SG_Total'])

  tournament_df['Round_Two_SG_Total'] = round(round_two_total_sg, 1)
  round_two_drop = tournament_df.dropna(subset=['Round_Two_SG_Total'])
  round_two_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_two_results = list(round_two_drop['Round_Two_SG_Total'])

  tournament_df['Round_Three_SG_Total'] = round(round_three_total_sg, 1)
  round_three_drop = tournament_df.dropna(subset=['Round_Three_SG_Total'])
  round_three_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_three_results = list(round_three_drop['Round_Three_SG_Total'])

  tournament_df['Round_Four_SG_Total'] = round(round_four_total_sg, 1)
  round_four_drop = tournament_df.dropna(subset=['Round_Four_SG_Total'])
  round_four_drop.drop_duplicates(subset=['Player'], inplace=True)
  round_four_results = list(round_four_drop['Round_Four_SG_Total'])

  # Return the final Stroke Gained total for the first two rounds and the final two rounds.
  first_two_rounds_results = [round(i + j, 1) for i, j in zip(round_one_results, round_two_results)]
  final_two_rounds_results = [round(x + y, 1) for x, y in zip(round_three_results, round_four_results)]

  # Create a list that does not contain duplicate player entries.
  clean_list = list(dict.fromkeys(list(tournament_df['Player'])))

  # Create a new Data Frame that contains Stroke Gained for each player.
  new_tournament_df = tournament_df.iloc[:len(clean_list)].copy()
  new_tournament_df['Player'] = clean_list
  new_tournament_df['Projected_First_Two_Rounds_SG_Totals'] = first_two_rounds_results

  # Take the Strokes Gained total for the first two rounds to determine the top ten finishers.
  new_tournament_df['Recent_Form_SG'] = round(new_tournament_df['Projected_First_Two_Rounds_SG_Totals'], 1)
  top_10_threshold = new_tournament_df['Recent_Form_SG'].nlargest(10).min()

  new_tournament_df['Top_Ten_Finish'] = (new_tournament_df['Recent_Form_SG'] >= top_10_threshold).astype(int)
  new_tournament_df = new_tournament_df.drop_duplicates(subset='Player')

  # Include players' current season averages into the Data Frame as a baseline for make future predictions.
  new_tournament_df = pd.merge(new_tournament_df, season_average_sg_total_df[['Player', 'Season_Average_SG_Total']],
                         on='Player',
                         how='left')
  new_tournament_df[['Season_Average_SG_Total']] = new_tournament_df[['Season_Average_SG_Total']].fillna(0.0)
  new_tournament_df.drop(columns=['Rounds', 'SG Total', 'Scrambling', 'Round_Three_SG_Total',
    'Round_Four_SG_Total'], inplace=True)

  new_tournament_df.to_csv('pga_data/sg_totals_with_top_performers.csv', index=False)

  # Prepare and train the model by including season Strokes Gained total stats, SG totals for the first two rounds, and top ten finishs.
  X = new_tournament_df[['Round_One_SG_Total', 'Round_Two_SG_Total', 'Recent_Form_SG']]
  y = new_tournament_df['Top_Ten_Finish']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create the model.
  # Include parameters to avoid overfitting to develop a more accurate prediction.
  rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    class_weight='balanced',
    random_state=42)

  rf_model.fit(X_train, y_train)

  # Get probabilities for all classes (0 = Field, 1 = Top 10)
  # [:, 1] grabs only the probability of finishing in the Top 10
  probabilities = rf_model.predict_proba(X_test)[:, 1]

  test_indices = X_test.index

  # Create the Leaderboard
  leaderboard = pd.DataFrame({
      'Player': new_tournament_df.loc[test_indices, 'Player'],
      'Actual_Top_10': y_test,
      'Probability': probabilities
  }).sort_values(by='Probability', ascending=False)

  print(leaderboard)
  leaderboard.to_csv('pga_data/top_ten_probabilities.csv', index=False)

  # Predict and Evaluate
  y_pred = rf_model.predict(X_test)

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  # View Feature Importance
  importances = pd.DataFrame({
      'Metric': X.columns,
      'Importance': rf_model.feature_importances_
  }).sort_values(by='Importance', ascending=False)

  print("\nFeature Importances:")
  print(importances)

  # def predict_top_10_performance_player_skill():
pebble_data = pd.read_csv('pga_data/pebble_beach_stats.csv')
phoenix_open_data = pd.read_csv('pga_data/wm_phoenix_open_stats.csv')
genesis_invitational_data = pd.read_csv('pga_data/genesis_invitational_stats.csv')
cognizant_class_data = pd.read_csv('pga_data/cognizant_classic_stats.csv')
arnold_palmer_data = pd.read_csv('pga_data/arnold_palmer_stats.csv')
players_championships_data = pd.read_csv('pga_data/players_championships_stats.csv')
valspar_championship_data = pd.read_csv('pga_data/valspar_championship_stats.csv')
texas_childrens_houston_open_data = pd.read_csv('pga_data/texas_childrens_houston_open_stats.csv')
valero_texas_open_data = pd.read_csv('pga_data/valero_texas_open_stats.csv')
masters_data = pd.read_csv('pga_data/masters_stats.csv')
seasonal_golf_data = pd.read_csv('pga_data/seasonal_golf_stats.csv')
phoenix_df = pd.DataFrame(phoenix_open_data)
genesis_invitational_df = pd.DataFrame(genesis_invitational_data)
pebble_beach_df = pd.DataFrame(pebble_data)
cognizant_classic_df = pd.DataFrame(cognizant_class_data)
arnold_palmer_df = pd.DataFrame(arnold_palmer_data)
players_championships_df = pd.DataFrame(players_championships_data)
valspar_championship_df = pd.DataFrame(valspar_championship_data)
texas_childrens_houston_open_df = pd.DataFrame(texas_childrens_houston_open_data)
valero_texas_open_df = pd.DataFrame(valero_texas_open_data)
masters_df = pd.DataFrame(masters_data)
seasonal_golf_df = pd.DataFrame(seasonal_golf_data)
tournaments = [phoenix_df, genesis_invitational_df, cognizant_classic_df, pebble_beach_df,
              arnold_palmer_df, players_championships_df, valspar_championship_df,
              texas_childrens_houston_open_df, valero_texas_open_df]
tournament_df = pd.concat(tournaments, ignore_index=True)

  # Calculate a player's current average Strokes Gained statistics per round.
round_one_total_sg_putting = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Putting'].transform('mean')
round_one_total_sg_around_green = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Around Green'].transform('mean')
round_one_total_sg_approach = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Approach'].transform('mean')
round_one_total_sg_off_the_tee = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Off The Tee'].transform('mean')
round_one_total_sg_tee_to_green = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Tee To Green'].transform('mean')
round_one_total_sg_total = tournament_df[tournament_df['Rounds'] == 1].groupby('Player')['SG Total'].transform('mean')

tournament_df['SG_Putting_Total_One'] = round(round_one_total_sg_putting, 3)
sg_one_putting_df = tournament_df.dropna(subset=['SG_Putting_Total_One'])
sg_one_putting_df = sg_one_putting_df.drop_duplicates(subset=['Player'])
round_one_total_sg_putting_list = np.array(list(sg_one_putting_df['SG_Putting_Total_One']))

tournament_df['SG_Around_Green_Total_One'] = round(round_one_total_sg_around_green, 3)
sg_one_around_green_df = tournament_df.dropna(subset=['SG_Around_Green_Total_One'])
sg_one_around_green_df = sg_one_around_green_df.drop_duplicates(subset=['Player'])
round_one_total_sg_around_green_list = np.array(list(sg_one_around_green_df['SG_Around_Green_Total_One']))

tournament_df['SG_Approach_Total_One'] = round(round_one_total_sg_approach, 3)
sg_one_approach_df = tournament_df.dropna(subset=['SG_Approach_Total_One'])
sg_one_approach_df = sg_one_approach_df.drop_duplicates(subset=['Player'])
round_one_total_sg_approach_list = np.array(list(sg_one_approach_df['SG_Approach_Total_One']))

tournament_df['SG_Off_The_Tee_Total_One'] = round(round_one_total_sg_off_the_tee, 3)
sg_one_off_the_tee_df = tournament_df.dropna(subset=['SG_Off_The_Tee_Total_One'])
sg_one_off_the_tee_df = sg_one_off_the_tee_df.drop_duplicates(subset=['Player'])
round_one_total_sg_off_the_tee_list = np.array(list(sg_one_off_the_tee_df['SG_Off_The_Tee_Total_One']))

tournament_df['SG_Tee_To_Green_Total_One'] = round(round_one_total_sg_tee_to_green, 3)
sg_one_tee_to_green_df = tournament_df.dropna(subset=['SG_Tee_To_Green_Total_One'])
sg_one_tee_to_green_df = sg_one_tee_to_green_df.drop_duplicates(subset=['Player'])
round_one_total_sg_tee_to_green_list = np.array(list(sg_one_tee_to_green_df['SG_Tee_To_Green_Total_One']))

tournament_df['SG_Total_One'] = round(round_one_total_sg_total, 3)
sg_one_total_df = tournament_df.dropna(subset=['SG_Total_One'])
sg_one_total_df = sg_one_total_df.drop_duplicates(subset=['Player'])
round_one_total_sg_total_list = np.array(list(sg_one_total_df['SG_Total_One']))


round_two_total_sg_putting = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Putting'].transform('mean')
round_two_total_sg_around_green = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Around Green'].transform('mean')
round_two_total_sg_approach = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Approach'].transform('mean')
round_two_total_sg_off_the_tee = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Off The Tee'].transform('mean')
round_two_total_sg_tee_to_green = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Tee To Green'].transform('mean')
round_two_total_sg_total = tournament_df[tournament_df['Rounds'] == 2].groupby('Player')['SG Total'].transform('mean')

tournament_df['SG_Putting_Total_Two'] = round(round_two_total_sg_putting, 3)
sg_two_putting_df = tournament_df.dropna(subset=['SG_Putting_Total_Two'])
sg_two_putting_df = sg_two_putting_df.drop_duplicates(subset=['Player'])
round_two_total_sg_putting_list = np.array(list(sg_two_putting_df['SG_Putting_Total_Two']))

tournament_df['SG_Around_Green_Total_Two'] = round(round_two_total_sg_around_green, 3)
sg_two_around_green_df = tournament_df.dropna(subset=['SG_Around_Green_Total_Two'])
sg_two_around_green_df = sg_two_around_green_df.drop_duplicates(subset=['Player'])
round_two_total_sg_around_green_list = np.array(list(sg_two_around_green_df['SG_Around_Green_Total_Two']))

tournament_df['SG_Approach_Total_Two'] = round(round_two_total_sg_approach, 3)
sg_two_approach_df = tournament_df.dropna(subset=['SG_Approach_Total_Two'])
sg_two_approach_df = sg_two_approach_df.drop_duplicates(subset=['Player'])
round_two_total_sg_approach_list = np.array(list(sg_two_approach_df['SG_Approach_Total_Two']))

tournament_df['SG_Off_The_Tee_Total_Two'] = round(round_two_total_sg_off_the_tee, 3)
sg_two_off_the_tee_df = tournament_df.dropna(subset=['SG_Off_The_Tee_Total_Two'])
sg_two_off_the_tee_df = sg_two_off_the_tee_df.drop_duplicates(subset=['Player'])
round_two_total_sg_off_the_tee_list = np.array(list(sg_two_off_the_tee_df['SG_Off_The_Tee_Total_Two']))

tournament_df['SG_Tee_To_Green_Total_Two'] = round(round_one_total_sg_tee_to_green, 3)
sg_two_tee_to_green_df = tournament_df.dropna(subset=['SG_Tee_To_Green_Total_Two'])
sg_two_tee_to_green_df = sg_two_tee_to_green_df.drop_duplicates(subset=['Player'])
round_two_total_sg_tee_to_green_list = np.array(list(sg_two_tee_to_green_df['SG_Tee_To_Green_Total_Two']))

tournament_df['SG_Total_Two'] = round(round_one_total_sg_total, 3)
sg_two_total_df = tournament_df.dropna(subset=['SG_Total_Two'])
sg_two_total_df = sg_two_total_df.drop_duplicates(subset=['Player'])
round_two_total_sg_total_list = np.array(list(sg_two_total_df['SG_Total_Two']))

round_three_total_sg_putting = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Putting'].transform('mean')
round_three_total_sg_around_green = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Around Green'].transform('mean')
round_three_total_sg_approach = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Approach'].transform('mean')
round_three_total_sg_off_the_tee = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Off The Tee'].transform('mean')
round_three_total_sg_tee_to_green = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Tee To Green'].transform('mean')
round_three_total_sg_total = tournament_df[tournament_df['Rounds'] == 3].groupby('Player')['SG Total'].transform('mean')

tournament_df['SG_Putting_Total_Three'] = round(round_three_total_sg_putting, 3)
sg_three_putting_df = tournament_df.dropna(subset=['SG_Putting_Total_Three'])
sg_three_putting_df = sg_three_putting_df.drop_duplicates(subset=['Player'])
round_three_total_sg_putting_list = np.array(list(sg_three_putting_df['SG_Putting_Total_Three']))

tournament_df['SG_Around_Green_Total_Three'] = round(round_three_total_sg_around_green, 3)
sg_three_around_green_df = tournament_df.dropna(subset=['SG_Around_Green_Total_Three'])
sg_three_around_green_df = sg_three_around_green_df.drop_duplicates(subset=['Player'])
round_three_total_sg_around_green_list = np.array(list(sg_three_around_green_df['SG_Around_Green_Total_Three']))

tournament_df['SG_Approach_Total_Three'] = round(round_three_total_sg_approach, 3)
sg_three_approach_df = tournament_df.dropna(subset=['SG_Approach_Total_Three'])
sg_three_approach_df = sg_three_approach_df.drop_duplicates(subset=['Player'])
round_three_total_sg_approach_list = np.array(list(sg_three_approach_df['SG_Approach_Total_Three']))

tournament_df['SG_Off_The_Tee_Total_Three'] = round(round_three_total_sg_off_the_tee, 3)
sg_three_off_the_tee_df = tournament_df.dropna(subset=['SG_Off_The_Tee_Total_Three'])
sg_three_off_the_tee_df = sg_three_off_the_tee_df.drop_duplicates(subset=['Player'])
round_three_total_sg_off_the_tee_list = np.array(list(sg_three_off_the_tee_df['SG_Off_The_Tee_Total_Three']))

tournament_df['SG_Tee_To_Green_Total_Three'] = round(round_one_total_sg_tee_to_green, 3)
sg_three_tee_to_green_df = tournament_df.dropna(subset=['SG_Tee_To_Green_Total_Three'])
sg_three_tee_to_green_df = sg_three_tee_to_green_df.drop_duplicates(subset=['Player'])
round_three_total_sg_tee_to_green_list = np.array(list(sg_three_tee_to_green_df['SG_Tee_To_Green_Total_Three']))

tournament_df['SG_Total_Three'] = round(round_three_total_sg_total, 3)
sg_three_total_df = tournament_df.dropna(subset=['SG_Total_Three'])
sg_three_total_df = sg_three_total_df.drop_duplicates(subset=['Player'])
round_three_total_sg_total_list = np.array(list(sg_three_total_df['SG_Total_Three']))

round_four_total_sg_putting = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Putting'].transform('mean')
round_four_total_sg_around_green = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Around Green'].transform('mean')
round_four_total_sg_approach = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Approach'].transform('mean')
round_four_total_sg_off_the_tee = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Off The Tee'].transform('mean')
round_four_total_sg_tee_to_green = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Tee To Green'].transform('mean')
round_four_total_sg_total = tournament_df[tournament_df['Rounds'] == 4].groupby('Player')['SG Total'].transform('mean')

tournament_df['SG_Putting_Total_Four'] = round(round_four_total_sg_putting, 3)
sg_four_putting_df = tournament_df.dropna(subset=['SG_Putting_Total_Four'])
sg_four_putting_df = sg_four_putting_df.drop_duplicates(subset=['Player'])
round_four_total_sg_putting_list = np.array(list(sg_four_putting_df['SG_Putting_Total_Four']))

tournament_df['SG_Around_Green_Total_Four'] = round(round_four_total_sg_around_green, 3)
sg_four_around_green_df = tournament_df.dropna(subset=['SG_Around_Green_Total_Four'])
sg_four_around_green_df = sg_four_around_green_df.drop_duplicates(subset=['Player'])
round_four_total_sg_around_green_list = np.array(list(sg_four_around_green_df['SG_Around_Green_Total_Four']))

tournament_df['SG_Approach_Total_Four'] = round(round_three_total_sg_approach, 3)
sg_four_approach_df = tournament_df.dropna(subset=['SG_Approach_Total_Four'])
sg_four_approach_df = sg_four_approach_df.drop_duplicates(subset=['Player'])
round_four_total_sg_approach_list = np.array(list(sg_four_approach_df['SG_Approach_Total_Four']))

tournament_df['SG_Off_The_Tee_Total_Four'] = round(round_four_total_sg_off_the_tee, 3)
sg_four_off_the_tee_df = tournament_df.dropna(subset=['SG_Off_The_Tee_Total_Four'])
sg_four_off_the_tee_df = sg_four_off_the_tee_df.drop_duplicates(subset=['Player'])
round_four_total_sg_off_the_tee_list = np.array(list(sg_four_off_the_tee_df['SG_Off_The_Tee_Total_Four']))

tournament_df['SG_Tee_To_Green_Total_Four'] = round(round_one_total_sg_tee_to_green, 3)
sg_four_tee_to_green_df = tournament_df.dropna(subset=['SG_Tee_To_Green_Total_Four'])
sg_four_tee_to_green_df = sg_four_tee_to_green_df.drop_duplicates(subset=['Player'])
round_four_total_sg_tee_to_green_list = np.array(list(sg_four_tee_to_green_df['SG_Tee_To_Green_Total_Four']))

tournament_df['SG_Total_Four'] = round(round_three_total_sg_total, 3)
sg_four_total_df = tournament_df.dropna(subset=['SG_Total_Four'])
sg_four_total_df = sg_four_total_df.drop_duplicates(subset=['Player'])
round_four_total_sg_total_list = np.array(list(sg_four_total_df['SG_Total_Four']))

# Calculate the total SG stats
total_sg_putting = round_one_total_sg_putting_list + round_two_total_sg_putting_list + round_three_total_sg_putting_list + round_four_total_sg_putting_list
total_sg_around_green = round_one_total_sg_around_green_list + round_two_total_sg_around_green_list + round_three_total_sg_around_green_list + round_four_total_sg_around_green_list
total_sg_approach = round_one_total_sg_approach_list + round_two_total_sg_approach_list + round_three_total_sg_approach_list + round_four_total_sg_approach_list
total_sg_off_the_tee = round_one_total_sg_off_the_tee_list + round_two_total_sg_off_the_tee_list + round_three_total_sg_off_the_tee_list + round_four_total_sg_off_the_tee_list
total_sg_tee_to_green = round_four_total_sg_tee_to_green_list + round_two_total_sg_tee_to_green_list + round_three_total_sg_tee_to_green_list + round_four_total_sg_tee_to_green_list
total_sg_total = round_four_total_sg_total_list + round_two_total_sg_total_list + round_three_total_sg_total_list + round_four_total_sg_total_list

np.set_printoptions(precision=3, suppress=True)
players = list(tournament_df['Player'].unique())
total_sg_putting = [round(x, 3) + 0.0 for x in total_sg_putting]
total_sg_around_green = [round(x, 3) + 0.0 for x in total_sg_around_green]
total_sg_approach = [round(x, 3) + 0.0 for x in total_sg_approach]
total_sg_off_the_tee = [round(x, 3) + 0.0 for x in total_sg_off_the_tee]
total_sg_tee_to_green = [round(x, 3) + 0.0 for x in total_sg_tee_to_green]
total_sg_total = [round(x, 3) + 0.0 for x in total_sg_total]
print(f"Unique players size: {len(players)}")
print(f"Putting size: {len(total_sg_putting)}")
print(f"Around green size: {len(total_sg_around_green)}")
sg_statistics = {
  'Player': players,
  'SG Putting': total_sg_putting,
  'SG Around Green': total_sg_around_green,
  'SG Approach': total_sg_approach,
  'SG Off The Tee': total_sg_off_the_tee,
  'SG Tee To Green': total_sg_tee_to_green,
  'SG Total': total_sg_total
}

players_sg_stats_df = pd.DataFrame(sg_statistics)

seasonal_golf_df['Player'] = seasonal_golf_df['Player'].str.lower()
seasonal_golf_df['Drive Accuracy'] = seasonal_golf_data['DACC'] / 100
players_sg_stats_df = players_sg_stats_df.merge(
    seasonal_golf_df[['Player', 'Drive Accuracy']],
    on='Player',
    how='left'
)

#  Fill any missing accuracy data with the average (so the model doesn't crash)
players_sg_stats_df['Drive Accuracy'] = players_sg_stats_df['Drive Accuracy'].fillna(
    players_sg_stats_df['Drive Accuracy'].mean()
)

threshold = players_sg_stats_df['SG Total'].nlargest(20).min()
players_sg_stats_df['Is_Top_20'] = (players_sg_stats_df['SG Total'] >= threshold).astype(int)
print(f"Data Frame: {players_sg_stats_df}")
print(f"Average sg total: {threshold}")

feature = [
  'SG Putting',
  'SG Around Green',
  'SG Approach',
  'SG Off The Tee'
]
# Prepare and train the model by including potential top 10 finishers.
X = players_sg_stats_df[feature]
y = players_sg_stats_df['Is_Top_20']

X = players_sg_stats_df[feature].copy()
X['SG Approach'] *= 1.8
X['SG Around Green'] *= 1.5
X['SG Putting'] *= 0.01
X['SG Off The Tee'] *= 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the model.
# Include parameters to avoid overfitting to develop a more accurate prediction.
rf_model = RandomForestClassifier(
  n_estimators=1000,
  max_depth=6,
  class_weight={0: 1, 1: 5},
  random_state=42)

# 1. Apply SMOTE only to the TRAINING data
sm = SMOTE(random_state=42, k_neighbors=2) # k_neighbors=2 because you have so few samples
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Winners in Training: {y_train_res.value_counts()[1]}")
print(f"Winners in Test: {y_test.value_counts().get(1, 0)}")
rf_model.fit(X_train_res, y_train_res)

# Get probabilities for all classes (0 = Field, 1 = Top 10)
# [:, 1] grabs only the probability of finishing in the Top 10
probabilities = rf_model.predict_proba(X_test)[:, 1]

test_indices = X_test.index
# 1. Prepare features for the ENTIRE dataset (not just the test split)
X_all = X.copy() 

# 2. Get probabilities for every single player in the file
all_probabilities = rf_model.predict_proba(X_all)[:, 1]

# Create the Leaderboard
leaderboard = pd.DataFrame({
    'Player': players_sg_stats_df['Player'],
    'Actual_Top_20': y,
    'Probability': all_probabilities
}).sort_values(by='Probability', ascending=False)

print(leaderboard)
leaderboard.to_csv('pga_data/top_ten_probabilities.csv', index=False)

# Predict and Evaluate
y_pred = (probabilities >= 0.25).astype(int)
print(f"Y train counts {y_train.value_counts()}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# View Feature Importance
importances = pd.DataFrame({
    'Metric': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importances)