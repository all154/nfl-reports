import nfl_data_py as nfl
import pandas as pd
import numpy as np

def situation():
    '''
        Slice data for each situation
    '''
    #TODO
    pass

def pass_rate_by_personnel(team, situation, side):
    '''
        Pass and rush rate split by personnel

        team (string): 3-letter abreviation of NFL team

        situation (string): different situation of a football match
            'COMP': all plays
            'OF': open field
            'RZ': red zone plays only
            '2M': 2 minute offense
            '4M': 4 minute offense
            'CLUT': clutch time (offense only - losing in the last 2 minutes)
        
        side (string): 
            'OFF': offensive personnel
            'DEF' for defensive personnel

        Returns: a dataframe with pass and rush rates by down, distance and situation
    '''
    pass

def pass_rate_by_formation():
    '''
    Description:
    '''
    pass

def pass_rate_by_personnel_and_formation():
    '''
    Description:
    '''
    pass