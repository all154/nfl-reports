import nfl_data_py as nfl
import pandas as pd
import numpy as np
from data_cleaner import *
from offense_tendencies import *

def situation(df, situation):
    '''
        Slice data for each situation in a football game

        df(dataframe): a play by play dataframe from nfl_data_py

        situation (string): different situation of a football match
            'COMP': all plays
            'OF': open field
            'RZ': red zone plays only
            '2M': 2 minute offense
            '4M': 4 minute offense
            'CLUT': clutch time (offense only - losing in the last 2 minutes)

        Returns: a dataframe with data with the situation given
    '''
    situations_dict = {
        'COMP': lambda df: df,
        'OF': lambda df: df[df['yardline_100'] > 20],
        'RZ': lambda df: df[df['yardline_100'] <= 20],
        '2M': lambda df: df[df['half_seconds_remaining'] <= 120],
        '4M': lambda df: df[(df['game_seconds_remaining'] <= 240) & (df['score_differential'] > 0)], 
        'CLUT': lambda df: df[(df['game_seconds_remaining'] <= 120) & (df['score_differential'] < 0)]
    }
    
    if situation in situations_dict:
        return situations_dict[situation](df)
    else:
        raise ValueError(f"Unknown situation: {situation}")

def team(df, team, side):
    '''
        Slice data for each team and side of a football game

        df(dataframe): a play by play dataframe from nfl_data_py

        team (string): 3-letter abreviation of NFL team

        side (string): 
            'OFF': offensive personnel
            'DEF' for defensive personnel

        Returns: a dataframe with data with the situation given
    '''
    side_dict = {
        'OFF': lambda df: df[(df['posteam'] == team)&(df['special'] == 0)],
        'DEF': lambda df: df[(df['defteam'] == team)&(df['special'] == 0)]
    }
    
    if side in side_dict:
        return side_dict[side](df)
    else:
        raise ValueError(f"Unknown side: {side}")
    
def import_clean_slice(years, weeks, team_name, side, situation_str):
    '''
        Description:
    '''
    df = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)
    df = create_distance(df)
    df = create_downs(df)
    df = clean_tendencies(df)
    df = team(df, team_name, side)
    df = situation(df, situation_str)

    return df

def order_pivot(pivot):
    '''
        Description:
    '''
    pivot = pivot.rename(columns={'distance': 'Play Count'})
    pivot = pivot.round(2)

    desired_order_down = {'P1st': 0, 'E1st': 1, '2nd': 2, '3rd': 3, '4th': 4}
    desired_order_distance = {'Long': 0, 'Medium': 1, 'Short': 2}

    pivot['Order_Down'] = pivot.index.get_level_values('Down').map(desired_order_down.get)
    pivot['Order_Distance'] = pivot.index.get_level_values('distance').map(desired_order_distance.get)
    pivot = pivot.sort_values(by=['Order_Down', 'Order_Distance']).drop(columns=['Order_Down', 'Order_Distance'])

    return pivot

def pass_rate_by_personnel(years, weeks, team_name, side, situation_str):
    '''
        Pass and rush rate split by personnel

        years (list): years that data should be retrieved

        weeks (list): weeks that data should be retrived

        team_name (string): 3-letter abreviation of NFL team

        side (string): 
            'OFF': offensive personnel
            'DEF' for defensive personnel

        situation_str (string): different situation of a football match
            'COMP': all plays
            'OF': open field
            'RZ': red zone plays only
            '2M': 2 minute offense
            '4M': 4 minute offense
            'CLUT': clutch time (offense only - losing in the last 2 minutes)

        Returns: a dataframe with pass and rush rates by down, distance and situation
    '''
    #TODO
    # selecting weeks and years for specific matchups

    side_dict = {
        'OFF': 'offense_personnel',
        'DEF': 'defense_personnel'
    }

    df = import_clean_slice(years, weeks, team_name, side, situation_str)

    pivot = pd.pivot_table(df, values=['pass','rush'], 
                                index=['Down', 'distance', side_dict[side]],
                                aggfunc={'pass': np.mean, 'rush': np.mean, 'distance': len},
                                fill_value=0)

    pivot = order_pivot(pivot)

    return pivot

def pass_rate_by_formation(years, weeks, team_name, situation_str):
    '''
    Description:
    '''
    df = import_clean_slice(years, weeks, team_name, 'OFF', situation_str)

    pivot = pd.pivot_table(df, values=['pass','rush'], 
                                index=['Down', 'distance', 'offense_formation'],
                                aggfunc={'pass': np.mean, 'rush': np.mean, 'distance': len},
                                fill_value=0)
    
    pivot = pivot.rename(columns={'distance': 'Play Count'})
    pivot = pivot.round(2)

    desired_order_down = {'P1st': 0, 'E1st': 1, '2nd': 2, '3rd': 3, '4th': 4}
    desired_order_distance = {'Long': 0, 'Medium': 1, 'Short': 2}

    pivot['Order_Down'] = pivot.index.get_level_values('Down').map(desired_order_down.get)
    pivot['Order_Distance'] = pivot.index.get_level_values('distance').map(desired_order_distance.get)
    pivot = pivot.sort_values(by=['Order_Down', 'Order_Distance']).drop(columns=['Order_Down', 'Order_Distance'])

    return pivot

def pass_rate_by_personnel_and_formation():
    '''
    Description:
    '''
    pass

def man_in_box_by_personnel():
    '''
        Description:
    '''
    pass