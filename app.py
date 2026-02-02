import services.nfl_services as nfl_services

nfl_services.get_nfl_team_offense_stats()
nfl_services.get_nfl_player_snap_count()
nfl_services.get_nfl_team_defense_stats()
nfl_services.get_overall_weighted_defensive_average()
nfl_services.get_nfl_fantasy_points_leaders()
nfl_services.get_nfl_dk_salary_data()
nfl_services.get_nfl_odds()
nfl_services.get_nfl_teams()
print(f"Get projections: {nfl_services.get_nfl_fantasy_projections()}")
# print(f"Get mascot names: {nfl_services.get_mascot_name()}")