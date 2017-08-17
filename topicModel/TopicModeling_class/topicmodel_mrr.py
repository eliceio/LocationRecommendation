from topicmodel_class_fix import TopicModel
import pandas as pd
import numpy as np # toy_data
import operator
import csv


import pdb

def load_data():
    '''
    Read data file
    File format: .csv, separated by tab
    '''

    # toy_data
    # df = pd.DataFrame(np.array([ [int(1), int(1), 0.2, 0.8], 
    #                            [int(2), int(2), 0.8, 0.2], 
    #                            [int(3), int(1), 0.2, 0.8], 
    #                            [int(3), int(2), 0.8, 0.2] ]), 
    #               columns = ['Member ID','Restaurant ID','Restaurant Latitude','Restaurant Longitude'])

    df = pd.read_csv('Daejeon_dataset.csv', delimiter='\t', index_col=False)
    return df

def cut_data(df, log_min, log_max):
    '''
    log_min, log_max에 맞추어 data 자르기
    잘라서, 축소된 df형태로 만드는 것이 목표
    '''
    df_user = df[['Member ID', 'Restaurant ID']]
    df_user = df_user.sort_values('Member ID')
    
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
    user_log_num = user_log_num[idx_min[0]:idx_max[0]-1] # user index has to start from 1
                                                         # [(user index, # of logs)]
    user_log_num = [list(x) for x in user_log_num]
    user_log_num = np.array(user_log_num)

    user_log_index = user_log_num[:,0] # 여기 있는 number들을 'Member ID'로 하는 값만 추리기
    
    cut_index = []
    training_data = []
    for mem_id in user_log_index.tolist():
        temp = df[df['Member ID']==mem_id]
        cut_index += temp.index.tolist()
        training_data.append(temp['Restaurant Name'].tolist())

    # pdb.set_trace()
    return user_log_index, df.loc[cut_index], training_data # training_data는 확인용

def separate_data(user_index, df):
   # pdb.set_trace()
   train_idx = []
   test_idx = []
   current_location_test = []

   for user_idx in user_index:
      df_idx = df[df['Member ID']==user_idx].index.tolist() # DataFrame index
      train_idx += df_idx #df_idx[:-1]
      test_idx.append(df_idx[-1]) # 숫자 하나여서

      # user_log_L = np.array(df.loc[train_idx, ['Restaurant Latitude', 'Restaurant Longitude']])
      user_log_L = np.array(df.loc[df_idx[:-1], ['Restaurant Latitude', 'Restaurant Longitude']])
      current_location_test.append(np.mean(user_log_L, axis=0).tolist())

   # return df_train, df_test, current_location_test
   return df.loc[train_idx], df.loc[test_idx], current_location_test


############################################################################

df = load_data()
print("Complete load data")

log_min = int(input("Enter the minimum number of log:"))
log_max = int(input("Enter the maximum number of log:"))
user_log_index, df_cut, training_data = cut_data(df, log_min, log_max) # user_index는 test를 위한 data를 만들때 사용
print("Complete rearrange data")

# pdb.set_trace()
df_train, df_test, current_location_test = separate_data(user_log_index, df_cut)

N = len(df_train['Member ID'].unique())
I = len(df_train['Restaurant ID'].unique())

print("User: %d, Location: %d" %(N, I))



# MRR

test_data = df_test['Restaurant Name'].tolist()

list_mrr = []

for user_idx, current_coordinate in enumerate(current_location_test):
    
    res_idx = df[df['Restaurant Name'] == test_data[user_idx]].index

    list_res_idx = res_idx.values.tolist()
    list_res_idx = np.array(list_res_idx)
    list_res_idx = 1.0 / (list_res_idx+1)
 

    mrr = np.sum(list_res_idx) / len(list_res_idx)
    list_mrr.append(mrr)


# mean MRR
print(np.mean(list_mrr))

