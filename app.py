import service.nfl_service as nfl_service
import service.pga_service as pga_service
import service.mlb_service as mlb_service

nfl_service.get_nfl_team_offense_stats()
nfl_service.get_nfl_player_snap_count()
nfl_service.get_nfl_team_defense_stats()
nfl_service.get_overall_weighted_defensive_average()
nfl_service.get_nfl_odds()
nfl_service.get_nfl_teams()
pga_service.player_expected_score_at_course()
pga_service.predict_top_10_performance()
# mlb_service.get_mlb_pitcher_profile()
# mlb_service.get_mlb_batter_profile()
# mlb_service.get_mlb_pitcher_national_averages()
# mlb_service.get_mlb_batter_national_averages()
mlb_service.generate_mlb_lineup()
# mlb_service.get_mlb_park_stats()
# mlb_service.generate_mlb_lineup_including_ballpark_factors()
# mlb_service.generate_mlb_lineup_including_hitter_ballpark_factors()
