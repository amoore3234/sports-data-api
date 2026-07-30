import services.nfl_service as nfl_service
import services.pga_service as pga_service
import services.mlb_services as mlb_services

nfl_service.get_nfl_team_offense_stats()
nfl_service.get_nfl_player_snap_count()
nfl_service.get_nfl_team_defense_stats()
nfl_service.get_overall_weighted_defensive_average()
nfl_service.get_nfl_odds()
nfl_service.get_nfl_teams()
nfl_service.generate_nfl_performance_probabilities()
pga_service.predict_pga_top_performers()
mlb_services.get_mlb_pitcher_profile()
mlb_services.get_mlb_batter_profile()
mlb_services.get_mlb_pitcher_national_averages()
mlb_services.get_mlb_batter_national_averages()
mlb_services.get_mlb_park_stats()