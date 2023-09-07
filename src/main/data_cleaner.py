import nfl_data_py as nfl
import pandas as pd
import numpy as np

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

    df['distance'] = pd.cut(year_data['ydstogo'], [0, 3, 6, 100], labels=['Short', 'Medium', 'Long'])

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