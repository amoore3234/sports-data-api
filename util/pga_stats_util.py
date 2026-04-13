import pandas as pd

# def generate_advanced_pga_stats():
dk_salary_data = pd.read_csv('../pga_data/dk_salaries.csv')
dk_salary_df = pd.DataFrame(dk_salary_data)
golf_data = pd.read_csv('../pga_data/golf_data.csv', encoding='cp1252')
golf_data_df = pd.DataFrame(golf_data)
golf_data_df = golf_data_df.iloc[:, ~golf_data_df.columns.str.contains('Unnamed')]
golf_data_df = golf_data_df.replace(r'[\u2013\u2014]', 0, regex=True)
golf_data_df = golf_data_df.map(lambda x: x.title() if isinstance(x, str) else x)

player_names = []
sg_putting = []
sg_around_green = []
sg_approach = []
sg_off_the_tee = []
sg_tee_to_green = []
sg_total = []

round_one_results = list(golf_data_df['Round 1'])
round_two_results = list(golf_data_df['Round 2'])
round_three_results = list(golf_data_df['Round 3'])
round_four_results = list(golf_data_df['Round 4'])

index = 0
count = 0
while index < len(round_one_results):
  player_name = round_one_results[index + 1]
  player_name_array = player_name.split()
  if player_name != 'Cut':
    player_first_last = f"{player_name_array[1]} {player_name_array[0]}"
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

    while count < 4:
      player_names.append(player_first_last)
      count += 1
  index+=10
  if player_name == 'Cut':
    index += 1
  count = 0
print(f"Player names: {player_names}")
print(f"SG Putting: {sg_putting}")
print(f"SG Around Green: {sg_around_green}")



