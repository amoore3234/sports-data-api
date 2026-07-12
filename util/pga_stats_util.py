import numpy as np
import pandas as pd

def get_seasonal_pga_data(is_seasonal) -> list[str]:
  """Generate historical PGA statistics

    Parameters:
      is_seasonal: Boolean value to toggle between historical and recent statistics.

    Returns:
      list[str]: List of strings
  """
  tournaments = get_tournaments()

  if not is_seasonal:
    tournaments = tournaments[5:]

  if len(tournaments) > 0:
    historic_game_stats = []
    for tournament in tournaments:
      data = pd.read_csv(f"pga_data/{tournament}.csv")
      df = pd.DataFrame(data)
      historic_game_stats.append(df)
  else:
    print('No data to read.')
  return historic_game_stats

def get_tournaments():
  return [
    'pebble_beach_stats',
    'wm_phoenix_open_stats',
    'genesis_invitational_stats',
    'cognizant_classic_stats',
    'arnold_palmer_stats',
    'players_championships_stats',
    'valspar_championship_stats',
    'texas_childrens_houston_open_stats',
    'valero_texas_open_stats',
    'masters_stats',
    'rbc_heritage_stats',
    'cadillac_championship',
    'truist_championship',
    'pga_championship',
    'cj_cup_byron_nelson_stats',
    'charles_schwab_challenge_stats',
    'the_memorial_tournament_stats',
    'rbc_canadian_stats',
    'us_open_stats',
    'travelers_championship_stats',
    'john_deere_stats'
  ]

def get_sg_statistics_average(df, round_number) -> dict:
  """Generate Strokes Gained (SG) statistics average per round.

    Parameters:
      df: Tournament DataFrame.
      round_number: Round value from a tournament.

    Returns:
      Dictionary: SG statistics.
  """
  round_average_sg_putting = df[df['Rounds'] == round_number].groupby('Player')['SG Putting'].transform('mean')
  round_average_sg_around_green = df[df['Rounds'] == round_number].groupby('Player')['SG Around Green'].transform('mean')
  round_average_sg_approach = df[df['Rounds'] == round_number].groupby('Player')['SG Approach'].transform('mean')
  round_average_sg_off_the_tee = df[df['Rounds'] == round_number].groupby('Player')['SG Off The Tee'].transform('mean')
  round_average_sg_tee_to_green = df[df['Rounds'] == round_number].groupby('Player')['SG Tee To Green'].transform('mean')
  round_average_sg_total = df[df['Rounds'] == round_number].groupby('Player')['SG Total'].transform('mean')
  round_average_scrambling = df[df['Rounds'] == round_number].groupby('Player')['Scrambling'].transform('mean')

  return {
    'round_average_sg_putting': round_average_sg_putting,
    'round_average_sg_around_green': round_average_sg_around_green,
    'round_average_sg_approach': round_average_sg_approach,
    'round_average_sg_off_the_tee': round_average_sg_off_the_tee,
    'round_average_sg_tee_to_green': round_average_sg_tee_to_green,
    'round_average_sg_total': round_average_sg_total,
    'round_average_scrambling': round_average_scrambling
  }

def get_player_recent_sg_statistics(df, is_seasonal) -> pd.DataFrame:
  """Get a player's current Strokes Gained (SG) statistics per round.

    Parameters:
      df: Tournament DataFrame
      is_seasonal: Boolean value to toggle between historical and recent statistics.

    Returns:
      DataFrame: SG statistics
  """
  df = get_recent_sg_statistics(df, 1, is_seasonal)
  df = get_recent_sg_statistics(df, 2, is_seasonal)
  df = get_recent_sg_statistics(df, 3, is_seasonal)
  df = get_recent_sg_statistics(df, 4, is_seasonal)

  return df

def get_recent_sg_statistics(df, round_number, is_seasonal) -> pd.DataFrame:
  """Get current or historical Strokes Gained (SG) data.

    Parameters:
      df: Tournament DataFrame.
      round_number: The round number value for a given tournament.
      is_seasonal: Boolean value to toggle between historical and current statistics.

    Returns:
      DataFrame: SG statistical data.
  """
  round_mask = df['Rounds'] == round_number

  if is_seasonal:
    stats_cols = [
      'seasonal_round_sg_putting_average',
      'seasonal_round_sg_around_green_average',
      'seasonal_round_sg_approach_average',
      'seasonal_round_sg_tee_to_green_average',
      'seasonal_round_sg_total_average',
      'seasonal_round_scrambling_average'
    ]
    df.rename(
      columns={
        'Player':'player',
        'SG Putting':'seasonal_round_sg_putting_average',
        'SG Around Green':'seasonal_round_sg_around_green_average',
        'SG Approach':'seasonal_round_sg_approach_average',
        'SG Tee To Green':'seasonal_round_sg_tee_to_green_average',
        'SG Off The Tee':'seasonal_round_sg_off_the_tee_average',
        'SG Total':'seasonal_round_sg_total_average',
        'Scrambling':'seasonal_round_scrambling_average'
      },
      inplace=True
    )
    df.loc[round_mask, stats_cols] = df[round_mask].groupby('player')[stats_cols].transform('mean')
  else:
    current_stats_cols = [
      'current_round_sg_putting',
      'current_round_sg_around_green',
      'current_round_sg_approach',
      'current_round_sg_tee_to_green',
      'current_round_sg_off_the_tee',
      'current_round_sg_total',
      'current_round_scrambling'
    ]
    df.rename(
      columns={
        'Player':'player',
        'SG Putting':'current_round_sg_putting',
        'SG Around Green':'current_round_sg_around_green',
        'SG Approach':'current_round_sg_approach',
        'SG Tee To Green':'current_round_sg_tee_to_green',
        'SG Off The Tee':'current_round_sg_off_the_tee',
        'SG Total': 'current_round_sg_total',
        'Scrambling': 'current_round_scrambling'
      },
      inplace=True
    )

    # Group all columns at once, calculate the mean, and assign back in place
    df.loc[round_mask, current_stats_cols] = df[round_mask].groupby('player')[current_stats_cols].transform('mean')

  return df