import pandas as pd
import numpy as np
import geopandas as gpd
import copy
import random
import json
import sys
pd.set_option('display.max_columns', None)


def progress_bar(process_rate):
    process = int(100*process_rate//1+1)
    print("\r", end="")
    print("Progress: {}%: ".format(process), "▋" * (process // 2), end="")
    sys.stdout.flush()


def fee(carry_distance):
    if carry_distance <= 2000:
        profit = 10 - cost_alpha * carry_distance
    if 2000 < carry_distance <= 20000:
        profit = 10 + 3.51 * (carry_distance - 2000) / 1000 - cost_alpha * carry_distance
    if 20000 < carry_distance <= 35000:
        profit = 10 + 3.51 * (carry_distance - 2000) / 1000 + 4.56 * (carry_distance - 20000) / 1000 - cost_alpha * carry_distance
    if carry_distance > 35000:
        profit = 10 + 3.51 * (carry_distance - 2000) / 1000 + 4.56 * (carry_distance - 20000) / 1000 + 5.62 * (carry_distance - 35000) / 1000 - cost_alpha * carry_distance
    return profit


def random_index(rate):
    rate_new = copy.deepcopy(rate)
    for i in range(0, len(rate_new)):
        rate_new[i] = 100000*rate[i]
    start = 0
    index = 0
    randnum = random.randint(1, int(sum(rate_new)))
    for index, scope in enumerate(rate_new):
        start += scope
        if randnum <= start:
            break
    return index


def simulation(grid_id, end_distance, vehicle_num):  # Output the average revenue and rate of vacant of the vehicle.
    current_grid_id = grid_id
    reward = 0
    distance = 0
    vacant_distance = 0
    carry_distance = 0
    reward_lst = []
    rate_lst = []
    n = 0
    while n < vehicle_num:
        while distance < end_distance:
            delta = end_distance - distance
            matching_prob = Matching_prob_df.loc[Matching_prob_df['current_grid_id'] == current_grid_id, 'matching_prob'].tolist()[0]
            no_matching_prob = 1-matching_prob
            matching_state = random_index([no_matching_prob, matching_prob])   # 根据概率随机确定是否接到乘客 0是未接到乘客，1是接到乘客
            df_neighbor =copy.deepcopy(Neighbour_df.groupby('current_grid_id').get_group(current_grid_id))
            df_neighbor.reset_index(inplace=True)
            neighbor_lst = list(pd.unique(df_neighbor['neighbor_grid_id']))
            if matching_state == 0:  # The next location is the grid corresponding to the action.
                next_grid_id = random.choice(neighbor_lst)
                if Grid_OD_distance_mtx[current_grid_id][next_grid_id] >= delta:
                    Dis = delta
                else:
                    Dis = Grid_OD_distance_mtx[current_grid_id][next_grid_id]
                reward = reward - cost_alpha*Dis
                distance = distance + Dis
                current_grid_id = copy.deepcopy(next_grid_id)
                vacant_distance = vacant_distance + Dis
            else:   # The next location is the grid where the passenger destination is located.
                if sum(list(Destination_prob_mtx[current_grid_id])) == 0:
                    next_grid_id = random_index(list([1 for i in range(0, len(Destination_prob_mtx[current_grid_id]))]))
                else:
                    next_grid_id = random_index(Destination_prob_mtx[current_grid_id])
                if Grid_OD_distance_mtx[current_grid_id][next_grid_id] >= delta:
                    Dis = delta
                else:
                    Dis = Grid_OD_distance_mtx[current_grid_id][next_grid_id]
                reward = reward + fee(Dis)
                distance = distance + Dis
                current_grid_id = copy.deepcopy(next_grid_id)
                carry_distance = carry_distance + Dis
        reward_lst.append(reward)
        rate_lst.append(carry_distance/(carry_distance+vacant_distance))
        n = n + 1
    average_reward = np.average(reward_lst)
    average_rate = np.average(rate_lst)
    return [average_reward/50, 1-average_rate]


print("Importing the grid division results... ...")
with open('D://状态升维马尔可夫计算（四叉树-大案例）//网格划分边界//0-1网格.json', 'r', encoding='utf8') as fp:
    grid_json = json.load(fp)
    Grid_gdf = gpd.GeoDataFrame.from_features(grid_json["features"])

print("Importing the distance between grids... ...")
fpath0 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Distance between grids.txt'
Grid_OD_distance_df = pd.read_csv(fpath0, sep=",", header=None, names=['from_grid_id', 'to_grid_id', 'distance'])

print("Importing the matching probabilities of grids... ...")
fpath1 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Matching probability.txt'
Matching_prob_df = pd.read_csv(fpath1, sep=",", header=None, names=['current_grid_id', 'matching_prob'])

print("Importing the destination probabilities of passengers... ...")
fpath2 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Destination probability.txt'
Destination_prob_df = pd.read_csv(fpath2, sep=",", header=None, names=['from_grid_id', 'to_grid_id', 'destination_prob'])

print("Importing the adjacency relationship of grids... ...")
fpath3 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Grid adjacency relationship.txt'  # 需要替换
Neighbour_df = pd.read_csv(fpath3, sep=",", header=None, names=['current_grid_id', 'neighbor_grid_id'])

print("Establishing the distance and destination probabilities matrix respectively... ...")
Destination_prob_mtx = np.zeros((len(Grid_gdf), len(Grid_gdf)))
for i in range(0, len(Destination_prob_df)):
    Destination_prob_mtx[Destination_prob_df.loc[i, 'from_grid_id']][Destination_prob_df.loc[i, 'to_grid_id']] = Destination_prob_df.loc[i, 'destination_prob']
Grid_OD_distance_mtx = 10000*np.ones((len(Grid_gdf), len(Grid_gdf)))
for i in range(0, len(Grid_OD_distance_df)):
    Grid_OD_distance_mtx[Grid_OD_distance_df.loc[i, 'from_grid_id']][Grid_OD_distance_df.loc[i,'to_grid_id']] = Grid_OD_distance_df.loc[i, 'distance']
for i in range(0,len(Grid_OD_distance_mtx)):  # Corrects to prevent the simulation from falling into a loop
    for j in range(0,len(Grid_OD_distance_mtx)):
        if Grid_OD_distance_mtx[i][j] <= 1000:
            Grid_OD_distance_mtx[i][j] = 1000

print("Importing and storing the vehicle initial simulation position... ...")
fpath4 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Vehicle initial position.txt'
Start_location_df = pd.read_csv(fpath4, sep=",", header=None, names=['grid_id', 'amount'])
start_id_lst = []
for i in range(0, len(Start_location_df)):
    if Start_location_df.loc[i, 'amount'] != 0:
        k = int(Start_location_df.loc[i, 'amount'])
    else:
        k = 0
    while k != 0:
        start_id_lst.append(Start_location_df.loc[i, 'grid_id'])
        k = k-1
print(start_id_lst)  # The set of vehicle initial simulation position'''
cost_alpha = 9.09*7.5/100000  # Unit distance (m) vehicle driving cost (No.92 gasolineJune, 30th, 2022)
distance_threshold = 50000  # 50km, Upper limit of single simulation driving distance of single vehicle
vehicle_numbers = 1  # Because the set “start_id_lst” has been used to set the initial position of every single vehicle
runs = 1000  # Number of simulation rounds

file_save = open('C://Users//张晨皓//Desktop//Simulation evaluation//result//0-1Random walk.txt', 'w').close()
file_save_handle = open('C://Users//张晨皓//Desktop//Simulation evaluation//result//0-1Random walk.txt', mode='a')
print("Starting the simulation evaluation... ...")
for i in range(0, runs):
    progress_bar(i/runs)
    reward_result = []
    rate_result = []
    for start_id in start_id_lst:
        reward_result.append(simulation(start_id, distance_threshold, vehicle_numbers)[0])
        rate_result.append(simulation(start_id, distance_threshold, vehicle_numbers)[1])
    print(np.average(reward_result), np.average(rate_result))
    file_save_handle.write(str(np.average(reward_result))+','+str(np.average(rate_result)))
    file_save_handle.write('\n')
file_save_handle.close()
