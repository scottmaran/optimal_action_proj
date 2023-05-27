import pandas as pd
import numpy as np
import time
import sys

def get_distances(x, player_pos):
    dist = round(np.linalg.norm(player_pos.values - x.values),3)
    return dist

try:
    year_str = str(sys.argv[1])
    print(f"Reading in CSV from year {year_str}")
    tracking = pd.read_csv(f"./data/tracking{year_str}.csv")
except:
    print(f"Did not get a valid year. Quiting")
    quit()

# get dataset of all Kickoff returns
play_data = pd.read_csv("./data/plays.csv")
returned_plays = play_data.query("specialTeamsResult == 'Return' and passResult.isnull() and penaltyYards.isnull()", engine='python', inplace=False)

start_time = time.time()

# intialize list of dictionaries (State, Action, Reward, New-state)
dict_list = []
num_plays = len(returned_plays)

print("Starting")
for index in range(0, num_plays):
    
    RECEIVED = False
    FUMBLED = False
    TACKLED = False

    play = returned_plays.iloc[index,].copy()
    gameId = play.loc['gameId']
    season = str(gameId)[0:4]
    playId = play.loc['playId']

    #####################
    # returnerId stored as decimal in tracking csv's
    try:
        returnerId = float(play.loc['returnerId'])
    except Exception  as e:
        print(F"Could not get returner id for index {index} at game {gameId} for play {playId}: {Exception}")
        continue
    
    # if no returner id, don't include play
    if pd.isnull(returnerId):
        continue
    
    play_df = tracking.query('playId == @playId and gameId == @gameId', inplace=False).copy()
    if len(play_df) == 0: 
        continue

    playDirection = play_df.iat[0,-1]
    
    if playDirection == 'left':
        play_df.loc[:, 'adj_x'] = play_df['x']
        play_df.loc[:, 'adj_y'] = play_df['y'] - 26.65
        # change dir and rotation
        play_df.loc[:, 'adj_o'] = (play_df['o'] + 90) % 360
        play_df.loc[:, 'adj_dir'] = (play_df['dir'] + 90) % 360

    if playDirection == 'right':
        play_df.loc[:, 'adj_x'] = 120 - play_df['x']
        play_df.loc[:, 'adj_y'] = 26.65 - play_df['y']
        # change dir and rotation
        play_df.loc[:, 'adj_o'] = (play_df['o'] - 90) % 360
        play_df.loc[:, 'adj_dir'] = (play_df['dir'] - 90) % 360
        
    play_df['sin_adj_o'] = np.sin(play_df['adj_o'])
    play_df['cos_adj_o'] = np.cos(play_df['adj_o'])
    play_df['sin_adj_dir'] = np.sin(play_df['adj_dir'])
    play_df['cos_adj_dir'] = np.cos(play_df['adj_dir'])

    play_df.fillna(0, inplace=True)

    # create next state
    play_df.loc[:, 'next_x'] = play_df.groupby('displayName').x.shift(-1)
    play_df.loc[:, 'next_y'] = play_df.groupby('displayName').y.shift(-1)
    play_df.loc[:, 'next_dis'] = play_df.groupby('displayName').dis.shift(-1)
    play_df.loc[:, 'next_s'] = play_df.groupby('displayName').s.shift(-1)
    play_df.loc[:, 'next_a'] = play_df.groupby('displayName').a.shift(-1)
    play_df.loc[:, 'next_o'] = play_df.groupby('displayName').o.shift(-1)
    play_df.loc[:, 'next_dir'] = play_df.groupby('displayName').dir.shift(-1)
    play_df.loc[:, 'next_adj_x'] = play_df.groupby('displayName').adj_x.shift(-1)
    play_df.loc[:, 'next_adj_y'] = play_df.groupby('displayName').adj_y.shift(-1)
    play_df.loc[:, 'next_adj_o'] = play_df.groupby('displayName').adj_o.shift(-1)
    play_df.loc[:, 'next_adj_dir'] = play_df.groupby('displayName').adj_dir.shift(-1)
    play_df.loc[:, 'next_sin_adj_o'] = play_df.groupby('displayName').sin_adj_o.shift(-1)
    play_df.loc[:, 'next_cos_adj_o'] = play_df.groupby('displayName').cos_adj_o.shift(-1)
    play_df.loc[:, 'next_sin_adj_dir'] = play_df.groupby('displayName').sin_adj_dir.shift(-1)
    play_df.loc[:, 'next_cos_adj_dir'] = play_df.groupby('displayName').cos_adj_dir.shift(-1)

    # remove NAs created by shifting; i.e. remove plays where there is no next play
    play_df.dropna(axis=0, how='any', subset=['next_adj_x'], inplace=True)
    
    # create reward - defined as forward progress
    play_df['reward'] = play_df['next_adj_x'] - play_df['adj_x']
    
    # get starting time of play
    starting_time = min(play_df['time'])
    all_time_steps = play_df.loc[:,'time'].unique()
    
    for time_index, time_string in enumerate(all_time_steps):
        # now we can define the values for each row
        rowDict = {}
            
        try:
            time_play_df = play_df.query("time == @time_string", inplace=False).copy()
            
            ball_carrier_team = time_play_df.loc[time_play_df.nflId == returnerId, ['team']].values[0][0] 
                
            other_team_name = 'home'
            if ball_carrier_team == 'home':
                other_team_name = 'away'

            # get distance from ball carrier
            ball_carrier_state = time_play_df.loc[time_play_df.nflId == returnerId, ['x','y', 's', 'a', 'o', 'dir']]
            time_play_df['dist_from_ball_carrier'] = time_play_df.loc[:,['x','y']].apply(get_distances, axis=1,args=(ball_carrier_state.loc[:, ['x','y']],))

            time_play_df['ball_carrier_bool'] = time_play_df.nflId != returnerId
            time_play_df['team_index'] = time_play_df.team == ball_carrier_team
            time_play_df.sort_values(by=["ball_carrier_bool","team_index","dist_from_ball_carrier"], inplace=True, ignore_index=True)

            #### 
            #Check to make sure ball carrier has football
            #   i.e. 'football' isn't second row in df and distance is less than hyperparameter
            ####
            football_index = time_play_df.index[time_play_df.team == 'football']
            try:
                football_distance = time_play_df.loc[football_index, 'dist_from_ball_carrier'].iloc[0] 
            except:
                football_distance = time_play_df.loc[football_index, 'dist_from_ball_carrier']
            
            event = list(np.unique(time_play_df.event))[0]
            if event == 'kick_received' or event == 'punt_received':
                RECEIVED = True
            if event == 'tackle' or event == 'out_of_bounds' or event == 'touchdown' or event == 'fumbled':
                TACKLED = True
            
            if TACKLED or (not RECEIVED):
                    continue
            
            # store info
            states = time_play_df.loc[:,['x','y', 'dis', 's', 'a', 'o', 'dir', 'adj_x', 'adj_y', 'adj_o', 'adj_dir', 'sin_adj_o', 'cos_adj_o', 'sin_adj_dir', 'cos_adj_dir', 'dist_from_ball_carrier']].to_numpy()
            football_pos = time_play_df.loc[time_play_df.team == 'football',['x','y', 'dis', 's', 'a', 'o', 'dir', 'adj_x', 'adj_y', 'adj_o', 'adj_dir', 'sin_adj_o', 'cos_adj_o', 'sin_adj_dir', 'cos_adj_dir', 'dist_from_ball_carrier']].to_numpy()
            
            rowDict['season'] = season
            rowDict['gameId'] = gameId
            rowDict['playId'] = playId
            
            rowDict["playIndex"] = index
            rowDict["timeIndex"] = time_index
            rowDict["time"] = time_string
            rowDict["football_pos"] = football_pos
            # add current state
            rowDict['state'] = states
            # add future states
            next_state = time_play_df.loc[:,['next_x','next_y', 'next_dis', 'next_s', 'next_a', 'next_o', 'next_dir', 'next_adj_x', 'next_adj_y', 'next_adj_o', 'next_adj_dir', 'next_sin_adj_o', 'next_cos_adj_o', 'next_sin_adj_dir', 'next_cos_adj_dir']].to_numpy()
            rowDict['next_state'] = next_state
            # add reward
            try:
                rowDict['reward'] = time_play_df.loc[0,'reward']
            except:
                print("error getting reward")
                continue
            
            action = states[0,7:-1] - next_state[0,7:]    # all states except dist_from_ball

            rowDict['action'] = action.round(3)

            dict_list.append(rowDict)
        except Exception as e:
            print(f"random error: {e}")
            time.sleep(5)
    
    if index % 250 == 0:
        print(f"Play {index} out of {num_plays}")

        
full_train_df = pd.DataFrame(dict_list)

full_train_df.loc[:, 'next_action'] = full_train_df.groupby('playIndex').action.shift(-1)

full_train_df = full_train_df.dropna(axis=0, how='any', inplace=False)

end_time = time.time()
total_time = round(end_time - start_time, 3)

full_train_df.to_pickle(f"datasets/bc_dataset_{year_str}.pkl")

print(F"total time: {total_time}  seconds")