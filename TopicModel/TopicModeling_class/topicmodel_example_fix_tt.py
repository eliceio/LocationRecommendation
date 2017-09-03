from topicmodel_class_fix import TopicModel
import pandas as pd
import numpy as np 
import operator
import matplotlib.pyplot as plt

import pdb

def load_data():
	'''
	Read data file
	File format: .csv, separated by tab
	1) Articifial_data (using generate_toydata.py)
	2) MangoPlage_data  
	'''
	
	## Articifial_data
	'''
	User 7
	Location 10
	Beta 1 Topic 3
	log_min = 4
	log_max = 7
	''' 
	# df = pd.read_csv('toy_data_7_10.csv', delimiter='\t', index_col=False)
	
	## MangoPlage_data
	df = pd.read_csv('Daejeon_dataset_t.csv', delimiter='\t', index_col=False)
	return df

def cut_data(df, log_min, log_max):
	'''
	log_min, log_max에 맞추어 data 자르기
	잘라서, 축소된 df형태로 만드는 것이 목표 
	'''
	# pdb.set_trace()
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

	# pdb.set_trace()
	if idx_max == []: # idx_max가 []라는 이유는, 가지고 있는 데이터에 log_max보다 더 큰 log를 가지는 애가 없다는 말
		user_log_num = user_log_num[idx_min[0]:idx_min[-1]+1] # +1은 list의 slicing
	else:
		user_log_num = user_log_num[idx_min[0]:idx_max[0]] # user index has to start from 1
														 # [(user index, # of logs)]
	user_log_num = [list(x) for x in user_log_num]
	user_log_num = np.array(user_log_num)

	user_log_index = user_log_num[:,0] # 여기 있는 number들을 'Member ID'로 하는 값만 추리기
	user_log_index = sorted(user_log_index)

	# pdb.set_trace()
	cut_index = []
	training_data = []
	for mem_id in user_log_index: #.tolist():
		temp = df[df['Member ID']==mem_id]
		cut_index += temp.index.tolist()
		training_data.append(temp['Restaurant Name'].tolist())

	df_cut = df.loc[cut_index]
	df_cut = df_cut.sort_values('Member ID')

	# pdb.set_trace()
	return user_log_index, df_cut, training_data # training_data는 확인용

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
#########
# Train #
#########
print('Topic modeling with mangoplate data')

df = load_data()
print("Complete load data")

log_min = int(input("Enter the minimum number of log:"))
log_max = int(input("Enter the maximum number of log:"))
user_log_index, df_cut, training_data = cut_data(df, log_min, log_max) # user_index는 test를 위한 data를 만들때 사용
print("Complete rearrange data")

df_train, df_test, current_location_test = separate_data(user_log_index, df_cut)

N = len(df_train['Member ID'].unique())
I = len(df_train['Restaurant ID'].unique())

print("User: %d, Location: %d" %(N, I))

beta = float(input("Enter the beta value:")) #
Z = int(input("Enter the number of topic:")) #

maxiter = int(input("Enter the number of maxiter:"))

sys1 = TopicModel(df_train, df_test, beta, Z, N, I) # df_test는 test에 들어가는 log를 training에서 배제하기 위함

print('\n')
print('Start parameter training: press "c"')
print('\n')
pdb.set_trace()

beta, psi = sys1.trainParams(maxiter = maxiter)
print('*********************************************************************')
print('***********************Training Complete*****************************')
print('*********************************************************************')

print('\n')
print('Go to Test: press "c"')
# print('Save psi: np.save(\'psi\', psi)')
# print('Or quit: press "q"')
print('\n')
pdb.set_trace()


#########
# Test ##
#########
## accuracy 측정
# user의 log 위치들의 중간 점에서, 추천을 하고, 그 가게가 추천한 것의 5개 안에 들어가는지로 판단

## input test data
## 새로운 장소에 해당 
# current_location = input("Enter the current space:")
# current_coordinate = sys1.get_location(current_location)

test_data = df_test['Restaurant Name'].tolist() # 얘랑 맞아야 하는 것임 

## 각 방법 별 추천 (Random, Max, GeoTopic)
# 한번에 모든 장소에 대해 갈 확률로 추천을 받아서, precision@N과 MRR을 계산
recommend_geo = [] # N * num_recommend
recommend_ran = [] # N * num_recommend
recommend_max = [] # N * num_recommend

num_recommend = I

## random
location = df_train['Restaurant Name'].unique().tolist() # np.array
# # 어차피 랜덤하게 sorting 할 것이므로, 여기서 sort 필요 없음!

## max
max_location_idx = df_train['Restaurant ID'].value_counts().index.tolist() 
max_location = []
for idx in max_location_idx:
	temp = df_train[df_train['Restaurant ID']==idx]['Restaurant Name'].tolist()
	max_location.append(temp[0])
# 충격! 'Restaurant ID'랑 'Restaurant Name'이랑 unique 갯수가 다름! 근데 training할때는 ID사용 

for user_idx, current_coordinate in enumerate(current_location_test):
	print('recommend || user %d' %(user_idx+1))
	recommend_prob = sys1.test(current_coordinate, psi, beta)
	recommendation = sys1.find_recommendation2(recommend_prob, num=num_recommend)
	recommend_geo.append(recommendation[user_idx])

	# recommendation = sys1.find_recommendation_random(num=num_recommend)
	# recommend_ran.append(recommendation)
	recommend_ran.append(np.random.permutation(location).tolist())

	recommend_max.append(max_location)

##### Measure 
print('*********************************************************************')
print('****************************Results**********************************')
print('*********************************************************************')

recommend = np.array(recommend_geo) # GEO TOPIC에서 가져온 recommendation
pre_at_N_geo = sys1.pre_at_N(recommend, test_data)
MRR_geo = sys1.MRR(recommend, test_data)

recommend = np.array(recommend_ran) # RANDOM에서 가져온 recommendation
pre_at_N_ran = sys1.pre_at_N(recommend, test_data)
MRR_ran = sys1.MRR(recommend, test_data)

recommend = np.array(recommend_max) # MAX LOG에서 가져온 recommendation
pre_at_N_max = sys1.pre_at_N(recommend, test_data)
MRR_max = sys1.MRR(recommend, test_data)

print('MRR')
print('Geotopic:%f' %MRR_geo)
print('Random  :%f' %MRR_ran)
print('Maxlog  :%f' %MRR_max)

print('Precision@n Graph')
plot_x = range(1, len(pre_at_N_geo)+1)
# plt.plot(pre_at_N_geo, label='GEO')
# plt.plot(pre_at_N_ran, label='RAN')
# plt.plot(pre_at_N_max, label='MAX')
plt.plot(plot_x, pre_at_N_geo, 'g--', plot_x, pre_at_N_ran, 'r--', plot_x, pre_at_N_max, 'b--')
plt.xlabel('N')
plt.ylabel('Precision@N')
plt.show()

# Topic extraciton
print('*********************************************************************')
print('****************************TOPIC__**********************************')
print('*********************************************************************')
topic_info, topic_info_sub = sys1.topic_extraction()
print('If you want to check topic_information, press c')
print('Or press anything except c')
topic_info_button = input()
if topic_info_button == 'c':
	print(np.array(topic_info))
	print(np.array(topic_info_sub))


print('\n')
print('To exit, press c')
print('Or you can check some variables, ...')
pdb.set_trace()



## 현재 위치의 주소를 입력받아, 모든 user에게 추천
# # test
# recommend_prob = sys1.test(current_coordinate, psi, beta)

# # print result
# recommendation = sys1.find_recommendation(recommend_prob)
# print(recommendation)


### change the file name loaded 

### Daejeon_dataset_t.csv
# Comment, Member ID, Member Nickname, Rating, Restaurant Address, Restuarant ID 
# Restaurant Latitude, Restaurant Longitude, Restaurant Nme, Restaurant code, 
# Restaurant subcode, Time  

### beta, Z optimization 


## 최종 코드:
# [[0, 1],
#  [1, 0],
#  [0.5, 0.5]]
# [[-6, 6], [5, -5]]

## 바꾼 코드:
# [[0.012, 0.987],
#  [0.989, 0.010],
#  [0.174, 0.825]]
# [[-13, 13],[5, -5]]

## 원래 코드 :
# [[3.27e-06, 9.99e-01],
#  [8.75e-02, 9.12e-01],
#  [1.10e-04, 9.99e-01]]
# [[-8, 8],[5,-5]]

########################################################################
## precision@5 계산 코드 
# 어차피, I개의 장소에 대해 갈 확률로 계산을 하니까, 
# 한번에 다 I개까지 계산을 해서, precision@N 이랑 MRR을 계산
# accuracy = 0
# test_result = []

# for user_idx, current_coordinate in enumerate(current_location_test):
	# recommend_prob = sys1.test(current_coordinate, psi, beta)
	# recommendation = sys1.find_recommendation2(recommend_prob, num=5)
	# test_result.append(recommendation[user_idx]) # N * num

	# if test_data[user_idx] in recommendation[user_idx]:
		# accuracy += 1

# accuracy = accuracy/len(test_data)*100
# print("accuracy is %f" %accuracy) # 지금 location 전체를 다 추천에 사용하고 있으므로, 100이 나와야 정상

########################################################################

# ## Measure
# def pre_at_N(recommend, test_data, I, N): # recommend는 numpy
# 	# recommend에서 i+1개씩 추천해서 accuracy 측정
# 	pre_at_N = []
# 	for i in range(I):
# 		recommend_at_N = recommend[:,0:(i+1)] # N * (i+1)

# 		accuracy = 0
# 		for n in range(N):
# 			if test_data[n] in recommend_at_N[n]:
# 				accuracy += 1
# 		accuracy = accuracy/N*100

# 		pre_at_N.append(accuracy)	
# 	return pre_at_N

# def MRR(recommend, test_data, N): # recommend는 numpy
# 	MRR = 0
# 	for n in range(N):
# 		t_recommend = recommend.tolist()
# 		mrr_index = t_recommend[n].index(test_data[n])+1
# 		print(mrr_index)
# 		MRR += 1/mrr_index
# 		MRR = MRR/N # 0.0366547
# 	return MRR

## Random 
# location = df_train['Restaurant Name'].unique().tolist() # np.array
# # 어차피 랜덤하게 sorting 할 것이므로, 여기서 sort 필요 없음!

# recommend = []
# for n in range(N):
# 	# np.random.shuffle(location) # deep copy ~하는 문제 발생
# 	recommend.append(np.random.permutation(location).tolist())
# recommend = np.array(recommend)