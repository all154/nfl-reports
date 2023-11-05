import nfl_data_py as nfl
import pandas as pd
import numpy as np

def clean_tendencies(df):
    '''
        Drop all features that will not be used for tendency studies

        df(dataframe): a play by play dataframe from nfl_data_py

        Returns: a dataframe with levant only features
    '''
    df = df.drop(['posteam_type', 'home_team', 'quarter_end', 'wpa', 'run_gap', 'away_team', 'old_game_id', 'drive_inside20',
                    'side_of_field', 'game_date', 'quarter_seconds_remaining', 'drive', 'sp', 'qtr', 'time', 'ydsnet', 'desc',
                    'yards_gained', 'qb_kneel', 'qb_spike', 'pass_length', 'pass_location', 'air_yards', 'yards_after_catch', 'run_location', 'field_goal_result',
                    'kick_distance', 'extra_point_result', 'two_point_conv_result', 'home_timeouts_remaining', 'away_timeouts_remaining', 'timeout',
                    'timeout_team', 'td_team', 'td_player_name', 'td_player_id', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
                    'total_home_score', 'total_away_score', 'posteam_score', 'defteam_score', 'posteam_score_post', 'defteam_score_post',
                    'score_differential_post', 'no_score_prob', 'opp_fg_prob', 'opp_safety_prob', 'opp_td_prob', 'fg_prob', 'safety_prob',
                    'td_prob', 'extra_point_prob', 'two_point_conversion_prob', 'ep', 'epa', 'total_home_epa', 'total_away_epa',
                    'total_home_rush_epa', 'total_away_rush_epa', 'total_home_pass_epa', 'total_away_pass_epa', 'air_epa', 'yac_epa',
                    'comp_air_epa', 'comp_yac_epa', 'total_home_comp_air_epa', 'total_away_comp_air_epa', 'total_home_comp_yac_epa',
                    'total_away_comp_yac_epa', 'total_home_raw_air_epa', 'total_away_raw_air_epa', 'total_home_raw_yac_epa', 'total_away_raw_yac_epa', 
                    'wp', 'def_wp', 'home_wp', 'away_wp', 'vegas_wpa', 'vegas_home_wpa', 'home_wp_post', 'away_wp_post', 'vegas_wp', 'vegas_home_wp', 
                    'total_home_rush_wpa', 'total_away_rush_wpa', 'total_home_pass_wpa', 'total_away_pass_wpa', 'air_wpa', 'yac_wpa', 'comp_air_wpa', 
                    'comp_yac_wpa', 'total_home_comp_air_wpa', 'total_away_comp_air_wpa', 'total_home_comp_yac_wpa', 'total_away_comp_yac_wpa', 
                    'total_home_raw_air_wpa', 'total_away_raw_air_wpa', 'total_home_raw_yac_wpa', 'total_away_raw_yac_wpa', 'punt_blocked', 
                    'first_down_rush', 'first_down_pass', 'first_down_penalty', 'third_down_failed', 'fourth_down_failed', 'incomplete_pass', 
                    'touchback', 'interception', 'punt_inside_twenty', 'punt_in_endzone', 'punt_out_of_bounds', 'punt_downed', 'punt_fair_catch', 
                    'kickoff_inside_twenty', 'kickoff_in_endzone', 'kickoff_out_of_bounds', 'kickoff_downed', 'kickoff_fair_catch', 'fumble_forced', 
                    'fumble_not_forced', 'fumble_out_of_bounds', 'solo_tackle', 'safety', 'penalty', 'tackled_for_loss', 'fumble_lost', 'own_kickoff_recovery', 
                    'own_kickoff_recovery_td', 'qb_hit', 'rush_attempt', 'pass_attempt', 'sack', 'touchdown', 'pass_touchdown', 'rush_touchdown', 
                    'return_touchdown', 'extra_point_attempt', 'two_point_attempt', 'field_goal_attempt', 'kickoff_attempt', 'punt_attempt', 'fumble', 
                    'complete_pass', 'assist_tackle', 'lateral_reception', 'lateral_rush', 'lateral_return', 'lateral_recovery', 'passer_player_id', 
                    'passer_player_name', 'passing_yards', 'receiver_player_id', 'receiver_player_name', 'receiving_yards', 'rusher_player_id', 
                    'rusher_player_name', 'rushing_yards', 'lateral_receiver_player_id', 'lateral_receiver_player_name', 'lateral_receiving_yards', 
                    'lateral_rusher_player_id', 'lateral_rusher_player_name', 'lateral_rushing_yards', 'lateral_sack_player_id', 'lateral_sack_player_name', 
                    'interception_player_id', 'interception_player_name', 'lateral_interception_player_id', 'lateral_interception_player_name', 
                    'punt_returner_player_id', 'punt_returner_player_name', 'lateral_punt_returner_player_id', 'lateral_punt_returner_player_name', 
                    'kickoff_returner_player_name', 'kickoff_returner_player_id', 'lateral_kickoff_returner_player_id', 'lateral_kickoff_returner_player_name', 
                    'punter_player_id', 'punter_player_name', 'kicker_player_name', 'kicker_player_id', 'own_kickoff_recovery_player_id', 
                    'own_kickoff_recovery_player_name', 'blocked_player_id', 'blocked_player_name', 'tackle_for_loss_1_player_id', 
                    'tackle_for_loss_1_player_name', 'tackle_for_loss_2_player_id', 'tackle_for_loss_2_player_name', 'qb_hit_1_player_id', 
                    'qb_hit_1_player_name', 'qb_hit_2_player_id', 'qb_hit_2_player_name', 'forced_fumble_player_1_team', 'forced_fumble_player_1_player_id', 
                    'forced_fumble_player_1_player_name', 'forced_fumble_player_2_team', 'forced_fumble_player_2_player_id', 'forced_fumble_player_2_player_name', 
                    'solo_tackle_1_team', 'solo_tackle_2_team', 'solo_tackle_1_player_id', 'solo_tackle_2_player_id', 'solo_tackle_1_player_name', 
                    'solo_tackle_2_player_name', 'assist_tackle_1_player_id', 'assist_tackle_1_player_name', 'assist_tackle_1_team', 'assist_tackle_2_player_id', 
                    'assist_tackle_2_player_name', 'assist_tackle_2_team', 'assist_tackle_3_player_id', 'assist_tackle_3_player_name', 'assist_tackle_3_team', 
                    'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 'assist_tackle_4_team', 'tackle_with_assist', 'tackle_with_assist_1_player_id', 
                    'tackle_with_assist_1_player_name', 'tackle_with_assist_1_team', 'tackle_with_assist_2_player_id', 'tackle_with_assist_2_player_name', 
                    'tackle_with_assist_2_team', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_defense_2_player_id', 'pass_defense_2_player_name', 
                    'fumbled_1_team', 'fumbled_1_player_id', 'fumbled_1_player_name', 'fumbled_2_player_id', 'fumbled_2_player_name', 'fumbled_2_team', 
                    'fumble_recovery_1_team', 'fumble_recovery_1_yards', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_2_team', 
                    'fumble_recovery_2_yards', 'fumble_recovery_2_player_id', 'fumble_recovery_2_player_name', 'sack_player_id', 'sack_player_name', 
                    'half_sack_1_player_id', 'half_sack_1_player_name', 'half_sack_2_player_id', 'half_sack_2_player_name', 'return_team', 'return_yards', 
                    'penalty_team', 'penalty_player_id', 'penalty_player_name', 'penalty_yards', 'replay_or_challenge', 'replay_or_challenge_result', 
                    'penalty_type', 'defensive_two_point_attempt', 'defensive_two_point_conv', 'defensive_extra_point_attempt', 'defensive_extra_point_conv', 
                    'safety_player_name', 'safety_player_id', 'season', 'cp', 'cpoe', 'series', 'series_success', 'series_result', 'order_sequence', 'start_time', 
                    'time_of_day', 'stadium', 'weather', 'nfl_api_id', 'play_clock', 'play_deleted', 'play_type_nfl', 'special_teams_play', 'st_play_type', 
                    'end_clock_time', 'end_yard_line', 'fixed_drive', 'fixed_drive_result', 'drive_real_start_time', 'drive_play_count', 'drive_time_of_possession', 
                    'drive_first_downs', 'drive_ended_with_score', 'drive_quarter_start', 'drive_quarter_end', 'drive_yards_penalized', 'drive_start_transition', 
                    'drive_end_transition', 'drive_game_clock_start', 'drive_game_clock_end', 'drive_start_yard_line', 'drive_end_yard_line', 
                    'drive_play_id_ended', 'away_score', 'home_score', 'location', 'result', 'total', 'spread_line', 'total_line', 
                    'div_game', 'roof', 'surface', 'temp', 'wind', 'home_coach', 'away_coach', 'stadium_id', 'game_stadium', 'aborted_play', 'success', 
                    'passer', 'passer_jersey_number', 'rusher', 'rusher_jersey_number', 'receiver', 'receiver_jersey_number', 'first_down', 'passer_id', 
                    'rusher_id', 'receiver_id', 'name', 'jersey_number', 'id', 'fantasy_player_name', 'fantasy_player_id', 'fantasy', 'fantasy_id', 
                    'out_of_bounds', 'home_opening_kickoff', 'qb_epa', 'xyac_epa', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success', 'xyac_fd', 
                    'xpass', 'pass_oe', 'nflverse_game_id', 'players_on_play', 'offense_players', 'defense_players', 'n_offense', 'n_defense',
                    'game_half', 'goal_to_go', 'play_type', 'qb_dropback', 'qb_scramble', 'third_down_converted', 'fourth_down_converted', 'possession_team'], axis='columns')
    
    return df

def create_distance(df):
    '''
        Group yards to go into 3 groups: Short, Medium and Long

        df(dataframe): a play by play dataframe from nfl_data_py containing 'ydstogo' feature

        Returns: a dataframe with 'distance' feature added
        
        Short: 1 to 3 yards to go
        Medium: 4 to 6 yards to go
        Long: More than 7 yards to go
    '''
    # TODO
    # Add check 'distance' feature

    df['distance'] = pd.cut(df['ydstogo'], [0, 3, 6, 100], labels=['Short', 'Medium', 'Long'])

    return df

def create_downs(df):
    '''
        Group downs into special groups

        df(dataframe): a play by play dataframe form nfl_data_py containing 'down' feature

        Returns: a dataframe with 'downs' feature added

        P1st: Possession first (first down of the drive)
        E1st: Earned first down
        2nd: Second down
        3rd: Third down
        4th: Fourth down
    '''
    #TODO
    #Check if there is 'down' feature
    
    df['Down'] = df['down'].astype(str)
    df['Down'] = df.apply(lambda row: 'P1st' if row['drive_play_id_started'] == row['play_id'] else row['Down'], axis=1)
    df['Down'] = df.apply(lambda row: 'E1st' if row['Down'] == '1.0' else row['Down'], axis=1)
    df['Down'] = df.apply(lambda row: '2nd' if row['Down'] == '2.0' else row['Down'], axis=1)
    df['Down'] = df.apply(lambda row: '3rd' if row['Down'] == '3.0' else row['Down'], axis=1)
    df['Down'] = df.apply(lambda row: '4th' if row['Down'] == '4.0' else row['Down'], axis=1)

    return df

def clean_offense_personnel(df):
    '''
        Usage of common personnel notation

        df(dataframe): a play by play dataframe from nfl_data_py containing 'offense_personnel' feature

        Returns: a dataframe with correct 'offense_personnel' notation

        First digit = # of RBs
        Second digit = # of TEs

        Example: 12 personnel = 1 RB, 2 TE, 2 WR

        Exceptions:
        Cougar = 00 personnel
        Y-Cougar = 01 personnel
        Jumbo = more than 5 OL
    '''
    #TODO
    pass

def create_explosive(df):
    '''
        Create explosive binary category

        df(dataframe): a play by play dataframe from nfl_data_py containing 'play_type' and 'yards_gained' feature

        Returns: a dataframe with 'explosive' feature

        Explosives:
        Run of more than 10 yards
        Pass of more than 15 yards
    '''
    #TODO
    pass

def create_negatives(df):
    '''
        Create negative (binary) category

        df(dataframe): a play by play dataframe from nfl_data_py containing 'play_type' and 'yards_gained' feature

        Returns: a dataframe with 'explosive' feature

        Negatives:
        Any play with loss of yards (TFL, sack or penalty)
    '''
    #TODO
    pass