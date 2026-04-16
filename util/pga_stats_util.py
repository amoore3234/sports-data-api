import math
import pandas as pd
import numpy as np

# def generate_advanced_pga_stats():
merge_data_set = False
golf_data = pd.read_csv('pga_data/golf_data.csv', encoding='cp1252')
golf_data_df = pd.DataFrame(golf_data)
golf_data_df = golf_data_df.iloc[:, ~golf_data_df.columns.str.contains('Unnamed')]
golf_data_df = golf_data_df.replace(r'[\u2013\u2014]', 0, regex=True)
golf_data_df = golf_data_df.replace('ï¿½', ' ')
golf_data_df = golf_data_df.map(lambda x: x.lower() if isinstance(x, str) else x)
golf_data_df = golf_data_df.replace(' ', 0)
name_map = {
  'min lee': 'min woo lee',
  'si kim': 'si woo kim',
  'matt mccarty': 'matthew mccarty',
  'iii capan': 'frankie capan iii',
  'rooyen van': 'erik van rooyen',
  'johnny keefer': 'john keefer',
  'zach bauchou': 'zachary bauchou',
  'de dumont': 'adrien dumont de chassart',
  'johnny keefer': 'john keefer',
  'zachary bauchou': 'zach bauchou',
  'nico echavarria': 'nicolas echavarria'
}

player_names = []
sg_putting = []
sg_around_green = []
sg_approach = []
sg_off_the_tee = []
sg_tee_to_green = []
sg_total = []
scrambling_statistic = []
ids = []
rounds = []

round_one_results = list(golf_data_df['Round 1'])
round_two_results = list(golf_data_df['Round 2'])
round_three_results = list(golf_data_df['Round 3'])
round_four_results = list(golf_data_df['Round 4'])

index = 0
count = 0
id = 0
rounds_count = 0
while index < len(round_one_results):
  player_name = round_one_results[index + 1]
  updated_name = player_name.replace('ï¿½', ' ')
  player_name_array = updated_name.split()
  if updated_name != 'cut':
    player_first_last = f"{player_name_array[1]} {player_name_array[0]}"
    for key, value in name_map.items():
      if key == player_first_last:
        player_first_last = value

    sg_putting.append(round_one_results[index + 4])
    sg_putting.append(round_two_results[index + 4])
    sg_putting.append(round_three_results[index + 4])
    sg_putting.append(round_four_results[index + 4])

    sg_around_green.append(round_one_results[index + 5])
    sg_around_green.append(round_two_results[index + 5])
    sg_around_green.append(round_three_results[index + 5])
    sg_around_green.append(round_four_results[index + 5])

    sg_approach.append(round_one_results[index + 6])
    sg_approach.append(round_two_results[index + 6])
    sg_approach.append(round_three_results[index + 6])
    sg_approach.append(round_four_results[index + 6])

    sg_off_the_tee.append(round_one_results[index + 7])
    sg_off_the_tee.append(round_two_results[index + 7])
    sg_off_the_tee.append(round_three_results[index + 7])
    sg_off_the_tee.append(round_four_results[index + 7])

    sg_tee_to_green.append(round_one_results[index + 8])
    sg_tee_to_green.append(round_two_results[index + 8])
    sg_tee_to_green.append(round_three_results[index + 8])
    sg_tee_to_green.append(round_four_results[index + 8])

    sg_total.append(round_one_results[index + 9])
    sg_total.append(round_two_results[index + 9])
    sg_total.append(round_three_results[index + 9])
    sg_total.append(round_four_results[index + 9])

    while count < 4:
      player_names.append(player_first_last)
      rounds.append(rounds_count + 1)
      ids.append(id + 1)
      scrambling_statistic.append(0)
      id += 1
      rounds_count += 1
      count += 1
  index+=10
  if player_name == 'cut':
    index += 1
  rounds_count = 0
  count = 0

sg_statistics = {
  'Player': player_names,
  'SG Putting': sg_putting,
  'SG Around Green': sg_around_green,
  'SG Approach': sg_approach,
  'SG Off The Tee': sg_off_the_tee,
  'SG Tee To Green': sg_tee_to_green,
  'SG Total': sg_total,
  'Rounds': rounds
}

sg_statistics_df = pd.DataFrame(sg_statistics)

if merge_data_set:
  data = pd.read_csv('pga_data/wm_phoenix_open_stats.csv')
  df = pd.DataFrame(data)
  df['Player'] = df['Player'].str.lower()

  statistics_df = sg_statistics_df.merge(
    df,
    left_on='Player',
    right_on='Player',
    how='left'
  )
  scrambling_list = list(statistics_df['Scrambling'])
  start = 0
  end = 4
  counter = 0
  new_scrambling = []
  while end < len(scrambling_list):
    stats = scrambling_list[start:end]
    while counter < len(stats):
      new_scrambling.append(scrambling_list[start])
      start += 1
      counter += 1
    counter = 0
    start += 12
    end += 16

  sg_statistics_df['Scrambling'] = new_scrambling
  
sg_statistics_df = sg_statistics_df.replace(r'^\s*$', np.nan, regex=True).fillna(0)
sg_statistics_df.to_csv('pga_data/masters_stats.csv', index=False)

