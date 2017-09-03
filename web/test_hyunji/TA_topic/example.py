
from GEOTOPIC import GeoTopic
import numpy as np
import pandas as pd
import operator

import matplotlib as mpl
mpl.use('Agg') 

import matplotlib.pyplot as plt

import pdb

def load_data():
    ## MangoPlage_data
    df = pd.read_csv('Daejeon_dataset.csv', delimiter='\t', index_col=False)
    df = df[['Member ID', 'Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude', 'Restaurant Name', 'Restaurant code', 'Restaurant subcode']]
    return df

def cut_data(df, log_min, log_max):
    '''
    log_min, log_max에 맞추어 data 자르기
    잘라서, 축소된 df형태로 만드는 것이 목표 
    '''
    df_user = df[['Member ID', 'Restaurant ID']] # 사용자별 log로 sorting하기 위해서
    # df_user.sort_values(['Member ID', 'Restaurant ID'], inplace=True)
    
    # visited_loc_user : x_um
    visited_loc_user = {}
    for index, row in df_user.iterrows():
        if row["Member ID"] not in visited_loc_user:
            visited_loc_user[row["Member ID"]] = [row["Restaurant ID"]]
        else:
            visited_loc_user[row["Member ID"]].append(row["Restaurant ID"])

    # 원하는 log 갯수만큼만 추리기 
    unique = np.array(list(visited_loc_user.keys()))
    counts = np.array([len(v) for v in visited_loc_user.values()])
    user_log_num = dict(zip(unique, counts))
    user_log_num = sorted(user_log_num.items(), key=operator.itemgetter(1))
    # for checkig data characteristic
    # np.save('user_log_num', np.array(user_log_num)) 

    idx_min = [idx for idx, item in enumerate(user_log_num) if item[1]>=log_min]
    idx_max = [idx for idx, item in enumerate(user_log_num) if item[1]>=log_max+1]

    if idx_max == []: # idx_max가 []라는 이유는, 가지고 있는 데이터에 log_max보다 더 큰 log를 가지는 애가 없다는 말
        user_log_num = user_log_num[idx_min[0]:idx_min[-1]+1] # +1은 list의 slicing
    else:
        user_log_num = user_log_num[idx_min[0]:idx_max[0]] # user index has to start from 1
                                                         # [(user index, # of logs)]
    user_log_num = [list(x) for x in user_log_num]
    user_log_num = np.array(user_log_num)

    user_log_index = user_log_num[:,0] # 여기 있는 number들을 'Member ID'로 하는 값만 추리기
    user_log_index = sorted(user_log_index)

    cut_index = []
    training_data = []
    for mem_id in user_log_index: #.tolist():
        temp = df[df['Member ID']==mem_id]
        cut_index += temp.index.tolist()
        training_data.append(temp['Restaurant Name'].tolist())

    df_cut = df.loc[cut_index]
    # df_cut.sort_values(['Member ID', 'Restaurant ID'], inplace=True)

    # pdb.set_trace()
    return user_log_index, df_cut, training_data
    # user_log_index: cut_data에서 해당하는 user, 작은 숫자부터 sorted.
    # df_cut: 잘려진 raw data, get_user_log에서 sorting할 것임. 
    # training_data: restaurant name으로만 이루어진 리스트 (test도 포함, 나중에 확인하기 위한 것임)

def separate_data(user_index, df):
    train_idx = []
    test_idx = []
    current_location_test = []

    # pdb.set_trace()
    for user_idx in user_index: # user_index는 sorting된 상태
        df_idx = df[df['Member ID']==user_idx].index.tolist() # DataFrame index
        train_idx += df_idx[:-1]
        test_idx.append(df_idx[-1]) # 숫자 하나여서

        # user_log_L = np.array(df.loc[train_idx, ['Restaurant Latitude', 'Restaurant Longitude']])
        user_log_L = np.array(df.loc[df_idx[:-1], ['Restaurant Latitude', 'Restaurant Longitude']])
        current_location_test.append(np.mean(user_log_L, axis=0).tolist())

    # pdb.set_trace()
    # return df_train, df_test, current_location_test
    return df.loc[train_idx], df.loc[test_idx], current_location_test
    # 이 return 값들은 모두 member id로 sorting

def get_user_log(df, df_train, df_test): 
    # 여기서는 장소는 전체 데이터가 필요하고, 
    # user_log는 train 데이터로만 되어야 함!
    uq_res_id = sorted(df['Restaurant ID'].unique()) # raw_data에서 sorting된 res id
    uq_mem_id = sorted(df['Member ID'].unique()) # raw_data에서 sorting된 mem id
    
    # df_train
    df_train = df_train[['Member ID', 'Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_train.sort_values(['Member ID', 'Restaurant ID'], inplace=True)
    # line97: A value is trying to be set on a copy of a slice from a DataFrame
    # See the caveats in the documentation: http://~ 

    data_tr = []
    for idx, row in df_train.iterrows():
        data_tr.append([uq_mem_id.index(row['Member ID'])+1, uq_res_id.index(row['Restaurant ID'])+1])
    data_tr = np.array(data_tr)

    # df_test
    df_test = df_test[['Member ID', 'Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_test.sort_values(['Member ID', 'Restaurant ID'], inplace=True)

    data_te = []
    for idx, row in df_test.iterrows():
        data_te.append([uq_mem_id.index(row['Member ID'])+1, uq_res_id.index(row['Restaurant ID'])+1])
    data_te = np.array(data_te)

    return data_tr, data_te

def get_log_info(df):
    uq_res_id = sorted(df['Restaurant ID'].unique())
    # uq_mem_id = sorted(df['Member ID'].unique())

    latlong = []
    for idx in uq_res_id:
        temp = df.loc[df['Restaurant ID']==idx].values[0]
        latlong.append([temp[2], temp[3]])

    latlong = np.array(latlong)

    # pdb.set_trace()
    return latlong


def data_to_csv(df_test, df, recommend_geo_cos):
    df.sort_values(['Member ID', 'Restaurant ID'], inplace=True)
    df = df.reset_index(drop=True)
    
    user = df_test['Member ID'].values.tolist()
    restaurant = []
    user_idx = 0
    
    for row in recommend_geo_cos:
        cnt=0
        for idx in row:
            if cnt == 5:
                break
            else:
                df_recm = df.ix[idx]
                # 'Restaurant ID', 'Restaurant Latitude','Restaurant Longitude', 'Restaurant Name',
                restaurant.append([user[user_idx],df_recm[1], df_recm[2], df_recm[3], df_recm[4]])
                cnt += 1

        user_idx += 1

    df_restaurant = pd.DataFrame(restaurant)
    df_restaurant.columns = ['Member ID','Restaurant ID', 'Restaurant Latitude','Restaurant Longitude', 'Restaurant Name']

    df_restaurant.to_csv('recommend_data.csv', index=False, encoding='utf-8')



## main
#################################################################################################
df = load_data() #pd.read_csv('Daejeon_dataset.csv', sep='\t', index_col=False)

log_min = int(input("Enter the minimum number of log:"))
log_max = int(input("Enter the maximum number of log:"))
user_log_index, df_cut, training_data = cut_data(df, log_min, log_max) # user_index는 test를 위한 data를 만들때 사용

df_train, df_test, current_location_test = separate_data(user_log_index, df_cut)

N = len(df_train['Member ID'].unique())
I = len(df_cut['Restaurant ID'].unique())

num_user = N
num_loca = I

print("User: %d, Location: %d" %(N, I))

beta = float(input("Enter the beta value:"))
num_topic = int(input("Enter the number of topic:"))
maxiter = int(input("Enter the number of maxiter:"))

user_log, test_data = get_user_log(df_cut, df_train, df_test) 
loc_info = get_log_info(df_cut) # location은 전체 다 있어야 함

### Create Model instance
sys1 = GeoTopic(num_user, num_loca, num_topic, beta, user_log, loc_info)

### Train Model
# Max iteration = maxiter
# Convergence threshold = 0.1

print('\n')
print('Start parameter training: press "c"')
print('\n')
pdb.set_trace()

sys1.trainParams(maxiter, 0.1)

print('*********************************************************************')
print('***********************Training Complete*****************************')
print('*********************************************************************')

# Print params
### In convergence, parameters would be
### Theta = [[0, 1], [1, 0], [0.5, 0.5]]
### Phi = [[postiive, negative], [negative, positive]]
# print(sys1.beta_dists)
# print(sys1.Theta)
# print(sys1.Phi)

print('\n')
print('Go to Test: press "c"')
print('\n')
pdb.set_trace()

####################################################################################################
### Recommend next locations

## RAN # list
recommend_ran = [np.random.permutation(np.array(range(num_loca))+1).tolist() for i in range(num_user)]

## MAX # list
locs = sorted(df_train['Restaurant ID'].unique())
data = []

for idx, loc in enumerate(locs):
    temp = df[df['Restaurant ID']==loc].shape
    data.append([loc, idx, temp[0]])

data_sorted = [row[1] for row in data] 
max_location = data_sorted
recommend_max = [max_location for i in range(num_user)]

## GEO # numpy array
user_id = np.array(range(1, num_user+1))
probs = sys1.recommendNext(user_id, current_location_test) # user의 각 location에서 추천한 확률값 
recommend_geo = np.flip(np.argsort(probs), axis=1)+1 

## GEO_SIM
recommend_geo_sim = []
for n in range(num_user):
    recommend_geo_sim.append(sys1.recommendNext_sim(n))

## GEO_SIM_COS
recommend_geo_cos = []
for n in range(num_user):
    recommend_geo_cos.append(sys1.recommendNext_cos(n))

print(df_test["Member ID"])
print(recommend_geo_cos)

print("----------result data to csv-------------")
#print(df_test)
data_to_csv(df_test, df_cut, recommend_geo_cos)


# print('*********************************************************************')
# print('****************************Results**********************************')
# print('*********************************************************************')
# recommend = np.array(recommend_ran) # RANDOM에서 가져온 recommendation
# pre_at_N_ran = sys1.pre_at_N(recommend, test_data)
# MRR_ran = sys1.MRR(recommend, test_data)

# recommend = np.array(recommend_max) # RANDOM에서 가져온 recommendation
# pre_at_N_max = sys1.pre_at_N(recommend, test_data)
# MRR_max = sys1.MRR(recommend, test_data)

# recommend = np.array(recommend_geo) # GEO TOPIC에서 가져온 recommendation
# pre_at_N_geo = sys1.pre_at_N(recommend, test_data)
# MRR_geo = sys1.MRR(recommend, test_data)

# recommend = np.array(recommend_geo_sim) # GEO TOPIC_SIM에서 가져온 recommendation
# pre_at_N_geo_sim = sys1.pre_at_N(recommend, test_data)
# MRR_geo_sim = sys1.MRR(recommend, test_data)

# recommend = np.array(recommend_geo_cos) # GEO TOPIC_SIM에서 가져온 recommendation
# pre_at_N_geo_cos = sys1.pre_at_N(recommend, test_data)
# MRR_geo_cos = sys1.MRR(recommend, test_data)

# # recommend = np.array(recommend_max) # MAX LOG에서 가져온 recommendation
# # pre_at_N_max = sys1.pre_at_N(recommend, test_data)
# # MRR_max = sys1.MRR(recommend, test_data)

# print('\n')
# print('MRR')
# print('Geotopic   :%f' %MRR_geo)
# print('GeotopicSim:%f' %MRR_geo_sim)
# print('GeotopicCos:%f' %MRR_geo_cos)
# print('Random     :%f' %MRR_ran)
# print('Maxlog     :%f' %MRR_max)

# print('Precision@N Graph')
# plot_x = range(1, len(pre_at_N_geo)+1)
# # plt.plot(pre_at_N_geo, label='GEO')
# # plt.plot(pre_at_N_ran, label='RAN')
# # plt.plot(pre_at_N_max, label='MAX')

# plt.plot(plot_x, pre_at_N_geo, 'g--', plot_x, pre_at_N_ran, 'r--', plot_x, pre_at_N_geo_sim, 'g', 
#     plot_x, pre_at_N_geo_cos, 'b--', plot_x, pre_at_N_max, 'o')
# # plt.plot(plot_x, pre_at_N_geo, 'g--', plot_x, pre_at_N_ran, 'r--', plot_x, pre_at_N_max, 'b--')
# plt.xlabel('N')
# plt.ylabel('Precision@N')
# #plt.show()

# plt.savefig('graph.png')

# pdb.set_trace()
