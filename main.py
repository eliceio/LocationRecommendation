
from TopicModel.GEOTOPIC import GeoTopic
import numpy as np
import pandas as pd
import operator

import matplotlib as mpl # to execute at SERVER
mpl.use('Agg')

import matplotlib.pyplot as plt

import pdb

def load_data():
    ## MangoPlage_data
    df = pd.read_csv('Data\Daejeon_dataset.csv', delimiter='\t', index_col=False)
    # pdb.set_trace()
    df = df[['Member ID', 'Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude', 'Restaurant Name', 'Restaurant code', 'Restaurant subcode']]

    # df = pd.read_csv('toy_data_7_10.csv', delimiter='\t', index_col=False)
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

def idx2name(lists):
    uq_res_id = sorted(df_cut['Restaurant ID'].unique())
    uq_mem_id = sorted(df_cut['Member ID'].unique())

    loc_name = []
    for i, list_idx in enumerate(lists):
        loc_name.append([])
        for idx in list_idx:
            temp = df_cut[df_cut['Restaurant ID']==uq_res_id[idx-1]]['Restaurant Name'].tolist()
            loc_name[i].append(temp[0])
    
    mem_name = []
    for idx in uq_mem_id:
        temp = df_cut[df_cut['Member ID']==idx]['Member Nickname'].tolist()
        # pdb.set_trace()
        mem_name.append(temp[0])    

    ##### 
    # 혹시 log가 많은 user들에 대해서 돌려서 결과를 얻고 싶으면, 
    # 이 자리에서 mem_name이랑 loc_name으로 data 얻으시면 됩니다~~
    # mam_name이랑 loc_name(각 user에게 추천된 가게, 정렬되어 있음)이랑 order는 같아요. 

    pdb.set_trace()
    return loc_name

## main
#################################################################################################
df = load_data() #pd.read_csv('Daejeon_dataset.csv', sep='\t', index_col=False)

log_min = 5#int(input("Enter the minimum number of log:"))
log_max = 5#int(input("Enter the maximum number of log:"))
user_log_index, df_cut, training_data = cut_data(df, log_min, log_max) # user_index는 test를 위한 data를 만들때 사용

df_train, df_test, current_location_test = separate_data(user_log_index, df_cut)

N = len(df_train['Member ID'].unique())
I = len(df_cut['Restaurant ID'].unique())

num_user = N
num_loca = I

print("User: %d, Location: %d" %(N, I))

beta = 10#float(input("Enter the beta value:"))
num_topic = 8#int(input("Enter the number of topic:"))
maxiter = 300#int(input("Enter the number of maxiter:"))

user_log, test_data = get_user_log(df_cut, df_train, df_test) 
loc_info = get_log_info(df_cut) # location은 전체 다 있어야 함

# pdb.set_trace()
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
# recommend_ 는 추천된 장소의 index (1부터 시작!)

## RAN # list
recommend_ran = [np.random.permutation(np.array(range(num_loca))+1).tolist() for i in range(num_user)]

## MAX # list
locs = sorted(df_cut['Restaurant ID'].unique())
data = []

for idx, loc in enumerate(locs):
    temp = df_cut[df_cut['Restaurant ID']==loc].shape ### df --> df_cut
    data.append([loc, idx+1, temp[0]])
    # [restaurant ID, index, log수] 
    # 장소의 index는 1부터 시작

data_sorted = sorted(data, key=lambda k:k[2], reverse=True) # log수로 sorting (작은게 앞)
#pdb.set_trace()
#data_sorted = data_sorted[::-1]
max_location = [row[1] for row in data_sorted] 
# max_location = data_sorted
recommend_max = [max_location for i in range(num_user)]

# pdb.set_trace()
## GEO # numpy array
user_id = np.array(range(1, num_user+1))
probs = sys1.recommendNext(user_id, current_location_test) # user의 각 location에서 추천한 확률값 
recommend_geo = np.flip(np.argsort(probs), axis=1)+1 
recommend_geo_re = recommend_geo[::-1]

## GEO_SIM
recommend_geo_sim = []
recommend_geo_sim_re = []
for n in range(num_user): # 행렬의 index를 넘겨 주니까 0부터
    recommend, recommend_re = sys1.recommendNext_sim(n)
    recommend_geo_sim.append(recommend)
    recommend_geo_sim_re.append(recommend_re)

# pdb.set_trace()
## GEO_SIM_COS
recommend_geo_cos = []
recommend_geo_cos_re = []
for n in range(num_user):
    recommend, recommend_re = sys1.recommendNext_cos(n)
    recommend_geo_cos.append(recommend)
    recommend_geo_cos_re.append(recommend_re)

# pdb.set_trace()
## GEO_SIM_Pearsonr
recommend_geo_pear = []
recommend_geo_pear_re = []
for n in range(num_user):
    recommend, recommend_re = sys1.recommendNext_pear(n)
    recommend_geo_pear.append(recommend)
    recommend_geo_pear_re.append(recommend_re)    
# pdb.set_trace()

#recommend_geo_name = idx2name(recommend_geo)
#recommend_geo_sim_name = idx2name(recommend_geo_sim)
#recommend_geo_cos_name = idx2name(recommend_geo_cos)
#recommend_geo_pear_name = idx2name(recommend_geo_pear)

#np.save('geo_name', np.array(recommend_geo_name))
#np.save('geo_sim_name', np.array(recommend_geo_sim_name))
#np.save('geo_cos_name', np.array(recommend_geo_cos_name))
#np.save('geo_pear_name', np.array(recommend_geo_pear_name))

# pdb.set_trace()

print('*********************************************************************')
print('****************************Results**********************************')
print('*********************************************************************')

pre_at_N_ran, MRR_ran = sys1.measure(np.array(recommend_ran), test_data)
pre_at_N_max, MRR_max = sys1.measure(np.array(recommend_max), test_data)

pre_at_N_geo, MRR_geo = sys1.measure(np.array(recommend_geo), test_data)
pre_at_N_geo_sim, MRR_geo_sim = sys1.measure(np.array(recommend_geo_sim), test_data)
pre_at_N_geo_cos, MRR_geo_cos = sys1.measure(np.array(recommend_geo_cos), test_data)
pre_at_N_geo_pear, MRR_geo_pear = sys1.measure(np.array(recommend_geo_pear), test_data)

pre_at_N_geo_re, MRR_geo_re = sys1.measure(np.array(recommend_geo_re), test_data)
pre_at_N_geo_sim_re, MRR_geo_sim_re = sys1.measure(np.array(recommend_geo_sim_re), test_data)
pre_at_N_geo_cos_re, MRR_geo_cos_re = sys1.measure(np.array(recommend_geo_cos_re), test_data)
pre_at_N_geo_pear_re, MRR_geo_pear_re = sys1.measure(np.array(recommend_geo_pear_re), test_data)


print('\n')
print('MRR')
print('Random     :%f' %MRR_ran)
print('Maxlog     :%f' %MRR_max)

print('Large/Small Probability')
print('Geotopic   :%f \t %f' %(MRR_geo, MRR_geo_re))
print('GeotopicSim:%f \t %f' %(MRR_geo_sim, MRR_geo_sim_re))
print('GeotopicCos:%f \t %f' %(MRR_geo_cos, MRR_geo_cos_re))
print('GeotopicPer:%f \t %f' %(MRR_geo_pear, MRR_geo_pear_re))

pdb.set_trace()
print('Precision@N Graph')
# plot_x = range(1, len(pre_at_N_geo)+1)
# plt.plot(pre_at_N_geo, label='GEO')
# plt.plot(pre_at_N_ran, label='RAN')
# plt.plot(pre_at_N_max, label='MAX')

# plt.plot(plot_x, pre_at_N_geo, 'g--', plot_x, pre_at_N_ran, 'r--', plot_x, pre_at_N_geo_sim, 'g', 
    # plot_x, pre_at_N_geo_cos, 'b--', plot_x, pre_at_N_max, 'o', plot_x, pre_at_N_geo_pear, 'k--')
# plt.plot(plot_x, pre_at_N_geo, 'g--', plot_x, pre_at_N_ran, 'r--', plot_x, pre_at_N_max, 'b--')
# plt.xlabel('N')
# plt.ylabel('Precision@N')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(pre_at_N_ran, label='RANDOM', linestyle='--', color='#525252')

ax1.plot(pre_at_N_geo, label='GEO', linestyle='--', color='#C8DEF9')
ax1.plot(pre_at_N_geo_sim, label='GEO_Euclidean', linestyle='--',color='#7EB2F1')
ax1.plot(pre_at_N_geo_cos, label='GEO_Cosine', linestyle='--', color='#A3C8F5')
ax1.plot(pre_at_N_geo_pear, label='GEO_Pearson', marker='o', color='#3586E9')

plt.xlabel('N')
plt.ylabel('Precision@N')

plt.legend(loc='upper left', frameon=False)
fig1.savefig('1.png')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.plot(pre_at_N_ran, label='RANDOM', linestyle='--', color='#474747')

ax2.plot(pre_at_N_geo_re, label='GEO_re', color='#B6D3F7')
ax2.plot(pre_at_N_geo_sim_re, label='GEO_Euclidean_re', color='#3586E9')
ax2.plot(pre_at_N_geo_cos_re, label='GEO_Cosine_re' , marker='o' , color='#1561BF')
ax2.plot(pre_at_N_geo_pear_re, label='GEO_Pearson_re',  color="#7EB2F1")

plt.xlabel('N')
plt.ylabel('Precision@N')

plt.legend(loc='upper left', frameon=False)
fig2.savefig('2.png')


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)

# ax1.plot(pre_at_N_ran, label='RANDOM', linestyle='--', color='k', marker='.')
# ax1.plot(pre_at_N_max, label='MAX', linestyle='--', color='b', marker='.')

# ax1.plot(pre_at_N_geo, label='GEO', linestyle='--', color='g', marker='.')
# ax1.plot(pre_at_N_geo_sim, label='GEO_Euclidean', linestyle='--', color='m', marker='.')
# ax1.plot(pre_at_N_geo_cos, label='GEO_Cosine', linestyle='--', color='c', marker='.')
# ax1.plot(pre_at_N_geo_pear, label='GEO_Pearson', linestyle='--', color='r', marker='o')

# plt.xlabel('N')
# plt.ylabel('Precision@N')
# plt.legend(loc='upper left', frameon=False)
# fig1.savefig('1.png')


# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)

# ax2.plot(pre_at_N_ran, label='RANDOM', linestyle='--', color='k', marker='.')
# ax2.plot(pre_at_N_max, label='MAX', linestyle='--', color='b', marker='.')

# ax2.plot(pre_at_N_geo_re, label='GEO', linestyle='--', color='g', marker='.')
# ax2.plot(pre_at_N_geo_sim_re, label='GEO_Euclidean', linestyle='--', color='m', marker='.')
# ax2.plot(pre_at_N_geo_cos_re, label='GEO_Cosine', linestyle='--', color='c', marker='o')
# ax2.plot(pre_at_N_geo_pear_re, label='GEO_Pearson', linestyle='--', color='r', marker='.')

# plt.xlabel('N')
# plt.ylabel('Precision@N')
# plt.legend(loc='upper left', frameon=False)
# fig2.savefig('2.png')

## 
# plt.plot(pre_at_N_geo_re, label='GEO_re')
# plt.plot(pre_at_N_geo_sim_re, label='GEO_Euclidean_re')
# plt.plot(pre_at_N_geo_cos_re, label='GEO_Cosine_re')
# plt.plot(pre_at_N_geo_pear_re, label='GEO_Pearson_re')

# plt.xlabel('N')
# plt.ylabel('Precision@N')

# plt.legend(loc='upper left')
plt.show()
# plt.savefig('graph.png')
# print('graph save')

print('\n')
print('If you want to finish, press "c"')
print('\n')

pdb.set_trace()
