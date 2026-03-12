import services.nfl_services as nfl_services
import services.pga_services as pga_services
import services.mlb_services as mlb_services

nfl_services.get_nfl_team_offense_stats()
nfl_services.get_nfl_player_snap_count()
nfl_services.get_nfl_team_defense_stats()
nfl_services.get_overall_weighted_defensive_average()
nfl_services.get_nfl_odds()
nfl_services.get_nfl_teams()
pga_services.player_expected_score_at_course()
pga_services.predict_top_10_performance()
# mlb_services.get_mlb_pitcher_profile()
# mlb_services.get_mlb_batter_profile()
# mlb_services.get_mlb_pitcher_national_averages()
# mlb_services.get_mlb_batter_national_averages()
mlb_services.generate_mlb_lineup()
# mlb_services.get_mlb_park_stats()
