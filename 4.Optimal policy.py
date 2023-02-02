# 靡不有初，鲜克有终
# 开发时间：2022/9/4 9:22
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


def simulation(grid_id, end_distance, vehicle_num):  # 输入车辆起点id、终止距离条件(米)、仿真次数；输出仿真之后，车辆的平均收益
    current_grid_id = grid_id
    previous_grid_id = grid_id
    reward = 0
    distance = 0
    total_vacant_distance = 0
    total_carry_distance = 0
    reward_lst = []
    rate_lst = []
    n = 0
    while n < vehicle_num:
        while distance < end_distance:
            delta = end_distance - distance
            matching_prob = Matching_prob_df.loc[Matching_prob_df['current_grid_id'] == current_grid_id, 'matching_prob'].tolist()[0]
            no_matching_prob = 1-matching_prob
            matching_state = random_index([no_matching_prob, matching_prob])   # 根据概率随机确定是否接到乘客 0是未接到乘客，1是接到乘客
            if matching_state == 0:  # 如果没有接到乘客，下一地点为动作对应的网格
                if len(Optimal_policy_df.loc[(Optimal_policy_df['current_grid_id'] == current_grid_id) & (Optimal_policy_df['previous_grid_id'] == previous_grid_id), 'to_grid_id'].tolist()) ==0:  # 考虑上一网格的策略里面没他
                    next_grid_id = Optimal_policy_df_0.loc[Optimal_policy_df_0['current_grid_id'] == current_grid_id, 'to_grid_id'].tolist()[0]
                    if Grid_OD_distance_3Dlst[current_grid_id][current_grid_id][next_grid_id] >= delta:
                        Dis = delta
                    else:
                        Dis = Grid_OD_distance_3Dlst[current_grid_id][current_grid_id][next_grid_id]
                    vacant_distance = Dis
                else:
                    next_grid_id = Optimal_policy_df.loc[(Optimal_policy_df['current_grid_id'] == current_grid_id) & (Optimal_policy_df['previous_grid_id'] == previous_grid_id), 'to_grid_id'].tolist()[0]
                    if Grid_OD_distance_3Dlst[previous_grid_id][current_grid_id][next_grid_id] >= delta:
                        Dis = delta
                    else:
                        Dis = Grid_OD_distance_3Dlst[previous_grid_id][current_grid_id][next_grid_id]
                    vacant_distance = Dis

                reward = reward - cost_alpha*vacant_distance  # 此时的reward只有成本（行驶距离）
                distance = distance + vacant_distance   # 累计距离
                previous_grid_id = copy.deepcopy(current_grid_id)
                current_grid_id = copy.deepcopy(next_grid_id)  # 车辆行驶到了下一个网格，所以将当前网格id用下一个网格id来替代，实现车辆移动
                total_vacant_distance = total_vacant_distance + vacant_distance

            else:   # 如果接到乘客，下一地点为乘客的目的地所在的网格
                if sum(list(Destination_prob_mtx[current_grid_id])) == 0:
                    next_grid_id = random_index(list([1 for i in range(0, len(Destination_prob_mtx[current_grid_id]))]))
                else:
                    next_grid_id = random_index(Destination_prob_mtx[current_grid_id])  # 根据乘客目的地概率随机确定下一个网格的index
                if Grid_OD_distance_3Dlst[previous_grid_id][current_grid_id][next_grid_id] >= delta:
                    Dis = delta
                else:
                    Dis = Grid_OD_distance_3Dlst[previous_grid_id][current_grid_id][next_grid_id]

                carry_distance = Dis
                reward = reward + fee(carry_distance)
                distance = distance + carry_distance
                previous_grid_id = copy.deepcopy(current_grid_id)
                current_grid_id = copy.deepcopy(next_grid_id)
                total_carry_distance = total_carry_distance + carry_distance
        reward_lst.append(reward)
        rate_lst.append(total_carry_distance/(total_carry_distance+total_vacant_distance))
        n = n + 1
    average_reward = np.average(reward_lst)
    average_rate = np.average(rate_lst)
    return [average_reward/50, 1-average_rate]  # 每千米的收益，空驶率


print("Importing the grid division results... ...")
with open('C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Grids.json', 'r', encoding='utf8') as fp:  # 需要替换
    grid_json = json.load(fp)
    Grid_gdf = gpd.GeoDataFrame.from_features(grid_json["features"])
Grid_OD_distance_3Dlst = [[[0 for k in range(0, len(Grid_gdf))]for j in range(0,len(Grid_gdf))]for i in range(0,len(Grid_gdf))]  # 'previous_grid_id', 'current_grid_id', 'to_grid_id','distance'

print("Importing the distance between grids... ...")
fpath4 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Distance between grids.txt'
Grid_OD_distance_df_0 = pd.read_csv(fpath4, sep=",", header=None, names=['from_grid_id', 'to_grid_id', 'distance'])
for i in range(0,len(Grid_OD_distance_df_0)):
    Y0 = Grid_OD_distance_df_0.loc[i, 'from_grid_id']
    Z0 = Grid_OD_distance_df_0.loc[i, 'to_grid_id']
    V0 = Grid_OD_distance_df_0.loc[i, 'distance']
    if V0 <= 1000:
        V0 = 1000
    for j in range(0, len(Grid_gdf)):
        Grid_OD_distance_3Dlst[j][Y0][Z0] = V0
fpath0 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Distance between grids 3D.txt'
Grid_OD_distance_df = pd.read_csv(fpath0, sep=",", header=None, names=['previous_grid_id', 'current_grid_id', 'to_grid_id','distance'])
for i in range(0, len(Grid_OD_distance_df)):
    X = Grid_OD_distance_df.loc[i, 'previous_grid_id']
    Y = Grid_OD_distance_df.loc[i, 'current_grid_id']
    Z = Grid_OD_distance_df.loc[i, 'to_grid_id']
    V = Grid_OD_distance_df.loc[i, 'distance']
current_grid_id_lst = list(pd.unique(Grid_OD_distance_df['current_grid_id']))

print("Importing the matching probabilities of grids... ...")
fpath1 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Matching probability.txt'
Matching_prob_df = pd.read_csv(fpath1, sep=",", header=None, names=['current_grid_id', 'matching_prob'])

print("Importing the destination probabilities of passengers... ...")
fpath2 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Destination probability.txt'
Destination_prob_df = pd.read_csv(fpath2, sep=",", header=None, names=['from_grid_id', 'to_grid_id', 'destination_prob'])

print("Establishing the distance and destination probabilities matrix respectively... ...")
Destination_prob_mtx = np.zeros((len(Grid_gdf), len(Grid_gdf)))
for i in range(0, len(Destination_prob_df)):
    Destination_prob_mtx[Destination_prob_df.loc[i, 'from_grid_id']][Destination_prob_df.loc[i, 'to_grid_id']] = Destination_prob_df.loc[i, 'destination_prob']

print("Importing the Optimal Policy... ...")
fpath3 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Optimal policy 3D.txt'  # 需要替换
Optimal_policy_df = pd.read_csv(fpath3, sep=",", header=None, names=['previous_grid_id', 'current_grid_id', 'to_grid_id'])
fpath5 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Optimal policy.txt'  # 需要替换
Optimal_policy_df_0 = pd.read_csv(fpath5, sep=",", header=None, names=['current_grid_id', 'to_grid_id'])

print("Establishing the distance and destination probabilities matrix respectively... ...")
Grid_OD_distance_mtx = 10000*np.ones((len(Grid_gdf), len(Grid_gdf)))
for i in range(0, len(Grid_OD_distance_df_0)):
    Grid_OD_distance_mtx[Grid_OD_distance_df_0.loc[i, 'from_grid_id']][Grid_OD_distance_df_0.loc[i,'to_grid_id']] = Grid_OD_distance_df_0.loc[i, 'distance']
for i in range(0,len(Grid_OD_distance_mtx)):
    for j in range(0,len(Grid_OD_distance_mtx)):
        if Grid_OD_distance_mtx[i][j] <= 1000:
            Grid_OD_distance_mtx[i][j] = 1000  # 1000米

print("Importing and storing the vehicle initial simulation position... ...")
fpath4 = 'C://Users//张晨皓//Desktop//Simulation evaluation//data//0-1Vehicle initial position.txt'
Start_location_df = pd.read_csv(fpath4, sep=",", header=None, names=['grid_id', 'amount'])
start_id_lst = []  # 用来储存车辆起始位置的网格id
for i in range(0, len(Start_location_df)):
    if Start_location_df.loc[i,'amount'] != 0:
        k = int(Start_location_df.loc[i,'amount'])
    else:
        k = 0
    while k != 0:
        start_id_lst.append(Start_location_df.loc[i,'grid_id'])
        k = k-1
# print(start_id_lst)  # 在start_id_lst去除非current_grid_id_lst的网格
start_id_lst_new = [0]
'''for i in start_id_lst:
    if i in current_grid_id_lst:
        start_id_lst_new.append(i)
# print(start_id_lst_new)  # 最终在start_id_lst去除非current_grid_id_lst之后的网格id集合'''

cost_alpha = 9.09*7.5/100000  # Unit distance (m) vehicle driving cost (No.92 gasolineJune, 30th, 2022)
distance_threshold = 50000  # 50km, Upper limit of single simulation driving distance of single vehicle
vehicle_numbers = 1  # Because the set “start_id_lst” has been used to set the initial position of every single vehicle
runs = 1000  # Number of simulation rounds

print("Starting the simulation evaluation... ...")
file_save = open('C://Users//张晨皓//Desktop//Simulation evaluation//result//0-1Optimal_profit.txt', 'w').close()
file_save_handle = open('C://Users//张晨皓//Desktop//Simulation evaluation//result//0-1Optimal_profit.txt', mode='a')
for i in range(0, runs):  # 分别依次将每个网格作为起点，运行50km则停止仿真，每个起点重复1次,总共进行1000轮
    progress_bar(i / runs)
    reward_result = []
    rate_result = []
    for start_id in start_id_lst_new:
        reward_result.append(simulation(start_id, distance_threshold, vehicle_numbers)[0])
        rate_result.append(simulation(start_id, distance_threshold, vehicle_numbers)[1])
    print(np.average(reward_result), np.average(rate_result))
    file_save_handle.write(str(np.average(reward_result))+','+str(np.average(rate_result)))
    file_save_handle.write('\n')
file_save_handle.close()