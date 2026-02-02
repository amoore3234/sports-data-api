import services.nfl_services as nfl_services

nfl_services.get_nfl_team_offense_stats()
nfl_services.get_nfl_player_snap_count()
nfl_services.get_nfl_team_defense_stats()
print('Yards per play allowed (weighted average):', nfl_services.get_overall_weighted_defensive_average())