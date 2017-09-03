'''
Author: Sumin Lim (KAIST), Hyunji Lee (Jeonbuk Univ.)
July 23th. 2017
Paper: Kurashima T. et al., Geo Topic Model: Joint Modeling of User's Acitivity Area and Interests for Location Recommendation
This program implements location recommendation using geotag data
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform # distance.euclidean(a,b)
import tensorflow as tf
from collections import Counter
import json
import urllib.request
from urllib import parse

import pdb

def E(psi, prob_loc_topic, df_user, visited_loc_user): 
		'''
		Expectation step for parameter estimation using EM algorithm
		In this step, the Bayse rule is used for computing the topic posterior probability
		Topic posterior probability refers to the probability of the mth location of user u gien the current estimate. 
		Input: 
		1. psi: [theta, phi]
		2. prob_loc_topic: dictionary, key = Member ID, value = Z*I dataFrame
		3. visited_loc_user: dictionary, key = Member ID, value = [restaurant ID, ... , restaurant ID]
		Output:
		Topic posterior probability: dictionary, key = Member ID, value = Z*I dataFrame
		'''
		np.seterr(divide='ignore', invalid='ignore')
		theta = psi[0]; phi = psi[1]; topic_prob = {}

		mem_Id = sorted(df_user['Member ID'].unique())
		loc_Id = sorted(df_user['Restaurant ID'].unique())

		theta = pd.DataFrame(theta, index=mem_Id)

		for key in visited_loc_user.keys(): # 마찬가지로 key는 user

				theta_usr = theta.loc[key] # 여긴 무조건 1*Z
				prob_loc_topic_usr = prob_loc_topic[key][visited_loc_user[key]]

				temp_col_name = prob_loc_topic_usr.columns
				prob_loc_topic_usr.columns = range(prob_loc_topic_usr.shape[1])

				prob_loc_topic_usr = prob_loc_topic_usr.multiply(theta_usr, axis=0)
				prob = prob_loc_topic_usr/prob_loc_topic_usr.sum()
				prob.columns = temp_col_name

				# 원래 코드랑 in/out을 맞추기 위해
				topic_prob[key] = pd.DataFrame((np.full([prob_loc_topic[key].shape[0], prob_loc_topic[key].shape[1]], np.nan)), columns=loc_Id)
				topic_prob[key].loc[:,prob.columns] = prob
				topic_prob[key] = topic_prob[key].fillna(0)
		
		return topic_prob

		
def get_ind(visited_loc_user, loc_Id):
		'''
		This function returns the indices for making tensor in M-step
		
		Input: 
		1. visited_loc_user: dictionary, key = Member ID, value = [restaurant ID, ... , restaurant ID]
		2. loc_Id: list, contains the unique restaurant IDs in the dataset
		Output:
		indices: nested list, which contains the user index and restaurant index. 
		User index refers to the index when sorting the dataset with Member ID. 
		For example, if member ID = 59, the index of user 59 is 0
		'''
		indices = []
		mem_Idx = -1
		for key in visited_loc_user.keys():
				mem_Idx += 1
				locs = visited_loc_user[key]
				for loc in locs:
						temp_loc = loc_Id.index(loc)
						indices.append([mem_Idx, temp_loc])

		return indices

def theta_optimize(prob_loc_topic): 
		'''
		This function returns the optimized theta in the M-step
		
		Input:
		The value of prob_loc_topic: Z * I dataFrame
		Output:
		theta_hat: optimized theta, N * Z
		'''
		theta_hat_numer = prob_loc_topic.sum(axis=1)
		theta_hat_denom = theta_hat_numer.sum()
		theta_hat = theta_hat_numer / theta_hat_denom
		return theta_hat

def M(visited_loc_user, topic_posterior_prob, Psi, distance, N, Z ,I, loc_Id): # half check
		'''
		This function is used for parameter estimation using EM algorithm
		Estimate theta, and maximize the conditional expectation of the complete-data log likelihood, equation (4) in the paper
		
		Input:
		1. visited_loc_user: the return value from the function get_visited_loc_user 
		2. topic_posterior_prob: the return value from the function E
		3. Psi: [theta, phi]
		4. distance: dataFrame
		5. N: the number of users
		6. Z: the number of topics
		7. I: the number of locations
		8. loc_id: list containing the restaurant ID
		Output:
		psi: [theta, phi]
		'''
		# Theta
		theta = []
		for key in topic_posterior_prob.keys():
				temp = theta_optimize(topic_posterior_prob[key])
				theta.append(temp)
		theta = np.array(theta)
		
		# Phi
		phi = Psi[1]
		phat = []

		for key in topic_posterior_prob.keys():
				phat.append(topic_posterior_prob[key].as_matrix())
				# phat: topic_posterior_prob (5)을 N*(Z*I) 형태의 np.array로 변형한 것.

		phat = np.array(phat)
		phat1 = np.swapaxes(phat, 0, 1)
		# phat의 정보를 topic별로, N*I로 변형.

		indices = get_ind(visited_loc_user, loc_Id)

		Indices = tf.SparseTensor(indices = indices, values = tf.ones(len(indices), dtype = tf.float64), dense_shape = [N, I])
		
		# placeholder
		Topic_post_prob = tf.placeholder(tf.float64, shape = [Z, N, I]) # shape ?/ M으로 가져온 topic_posterior_prob은 [N(3), Z, I]
		Theta = tf.placeholder(tf.float64, shape = [N, Z])
		Phi = tf.placeholder(tf.float64, shape = [Z, I])
		Dist = tf.placeholder(tf.float64, shape = [I, I])
		
		## 
		# 여기서 만약 self.prob_loc_topic이 있으면,
		# 얘가 그냥 아래 코드에서 P가 됨 --> 함수 M2, dataframe 변형이 어려워서 그냥 둠

		# Calculate P(visited_loc_user|z, R_u, Psi)
		# Z * I
		front = tf.exp(Phi)

		# N x I
		back = tf.sparse_tensor_dense_matmul(Indices, Dist)

		P_numer = tf.expand_dims(front, axis =1) * back # Z * N * I
		P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis = 2), axis = 2)
		P = P_numer / P_denom
		
		log_Theta = tf.expand_dims(tf.transpose(Theta), axis = 2) # 
		loglike = Topic_post_prob * tf.log(log_Theta * P) # log의 위치 수정, loglike checked.
		# The conditional expectation of the log likelihood
		# 변수 이름을 loglike라 지어놨지만, 사실상 Q를 구하는 과정임, 식(4)

		Q = tf.negative(tf.reduce_sum(loglike)) # Q 정의 수정. 다시 곱하지 않아도 됨. loglike의 합이 되어야 함.

		Phi_grad = tf.gradients(Q, Phi)

		sess = tf.Session()

		def objective(phi_):
				phi_ = phi_.reshape(Z, I)
				#phat_, theta_, phi_, distance_ = param
				feed_dict={Topic_post_prob: phat1, Theta: theta, Phi: phi_, Dist: distance}
				return sess.run(Q, feed_dict={Topic_post_prob: phat1, Theta: theta, Phi: phi_, Dist: distance})

		def gradient(phi_):
				phi_ = phi_.reshape(Z, I)
				#phat_, theta_, phi_, distance_ = param
				feed_dict={Topic_post_prob: phat1, Theta: theta, Phi: phi_, Dist: distance}
				ret = sess.run(Phi_grad, feed_dict={Topic_post_prob: phat1, Theta: theta, Phi: phi_, Dist: distance})
				res = np.squeeze(ret).flatten()
				return res

		res =  minimize(objective, x0=phi, jac=gradient)

		return [theta, res.x]


class TopicModel():

		def __init__(self, df, df_test, beta, Z, N, I):
				
				self.df = df # df_train
				self.df_test = df_test
				self.beta = beta
				self.Z = Z 
				self.N = N
				self.I = I

		def __initialize(self):
				# df_loc, df_user, loaction, loc_Id, mem_Id
				# df_dist, distance, visited_loc_user
				# L

				self.df_loc = self.df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]

				self.df_user = self.df[['Member ID', 'Restaurant ID']]

				self.location = sorted(list(set([tuple(x) for x in self.df_loc.to_records(index=False)])))

				self.L = np.array([[x[1], x[2]] for x in self.location])

				self.loc_Id = sorted(self.df_user['Restaurant ID'].unique())
				self.mem_Id = sorted(self.df_user['Member ID'].unique())
				
				# out: self.df_dist
				self.__getDist()
				self.distance = self.df_dist.as_matrix()

				# out: self.visited_loc_user
				self.__get_visited_loc_user() # self.visited_loc_user

				# out: self.psi
				self.__psi_initialize(std=.1) # self.psi

				# out: self.prob_loc_topic
				self.__get_prob_loc_topic() # self.prob_loc_topic 


		def __getDist(self): #checked
				'''
				Calculate the distance between locations 
				Input
				beta: scalar
				location: should be sorted list
				
				Output
				distance: dataframe, shape of I * I, row and column index = restaurant ID
				'''

				dist = np.exp(squareform(-0.5*self.beta*pdist(self.L, 'sqeuclidean')))
				dist = pd.DataFrame(dist, columns=self.loc_Id, index=self.loc_Id)

				self.df_dist = dist

		def __get_visited_loc_user(self): # checked
				'''
				Get the visited locations per user
				Input: DataFrame
				Output: Dictionary, key = Member ID, value = [restaurant ID, ... , restaurant ID]
				The length of each value differs from each user (len(value) = Mu)
				'''
				df_temp = self.df.sort_values('Member ID')
				df_user = df_temp[["Member ID", "Restaurant ID"]]

				location_counts = df_user.groupby("Member ID").size()
				
				# visited_loc_user : x_um
				visited_loc_user = {}
				for user_idx in self.mem_Id:
					temp = df_user[df_user['Member ID']==user_idx]['Restaurant ID']
					visited_loc_user[user_idx] = temp.tolist()

				for index, row in self.df_test.iterrows():
					visited_loc_user[row['Member ID']].remove(row['Restaurant ID'])
						
				self.visited_loc_user = visited_loc_user


		def __psi_initialize(self, std = .1): # checked
				'''
				Initialize theta and phi
				theta shape: N * Z
				phi shape: Z * I
				'''
				theta = np.random.rand(self.N, self.Z)
				row_sum = theta.sum(axis=1).reshape(-1, 1)
				theta = theta / row_sum
				# phi = np.random.rand(self.Z, self.I)
				phi = np.random.normal(loc = 0.0, scale = std, size = [self.Z, self.I])

				# Test
				# theta = np.array([[ 0.27608385,  0.72391615],
												  # [ 0.83381262,  0.16618738],
												  # [ 0.1910022 ,  0.8089978 ]])
				# phi = np.array([[-0.15911254,  0.02733004],
												# [ 0.07896183,  0.10809377]])

				self.psi = [theta, phi]

		def __get_prob_loc_topic(self): # checked
				'''
				P(i|z, R_u, Phi), equation (2) in the paper
				location i is chosen from topic z after consideration of the user's geotags R_u
				Input:
				location (dataFrame), df_dist (dataFrame), visited_loc_user (dictionary), psi ([theta, phi])
				Output:
				Dictionary with key = Member ID, value = Z * I dataFrame
				''' 
				loc_Id = np.array([x[0] for x in self.location])
				phi = self.psi[1]; phi = np.exp(phi)
				phi = pd.DataFrame(phi, columns=loc_Id)
				prob_loc_topic = {}; #I = len(loc_Id); Z = phi.shape[0]

				# ## 모든 user에게 안간 곳도 확률을 계산하도록 해야함.
				for key in self.visited_loc_user.keys():
						temp = np.full([self.Z, self.I], np.nan)
						df_temp = pd.DataFrame(temp, columns=loc_Id)
		
				# Eq.(2)의 거리 계산 부분을 위해, log를 가져옴
						usr_vsted_loc = self.visited_loc_user[key] # user별 방문 log
						user_dist = self.df_dist.loc[:,usr_vsted_loc]
						user_dist_sum = user_dist.sum(axis=1)
						
						# 여기부터는 index를 0번부터로 맞춰야 계산이 가능해서
						user_dist_sum = user_dist_sum.reset_index(drop=True)
						temp_col_name = phi.columns
						phi.columns = range(phi.shape[1])
						
						# 확률 계산
						prob = phi.multiply(user_dist_sum,axis=1) # 한 row씩 seriese랑 곱하기, Z * I
						prob = prob.div(prob.sum(axis=1), axis=0) # pXum을 계산할 때, 한 topic에서 모든 location에 대해 normalization

						# for를 다시 돌기 위해 columns 원래대로
						phi.columns = temp_col_name
						prob.columns = temp_col_name
								
						prob_loc_topic[key] = prob

				self.prob_loc_topic = prob_loc_topic


		def trainParams(self, maxiter):

				self.__initialize()

				niter = 0
				cnt = 0
				
				# print('Init')
				# print(self.psi[0])
				# print(self.psi[1])

				for niter in range(1, maxiter):
						cnt += 1
						print("==================== In the While Loop =======================")
						print(" %d th iteration" % cnt)

						pre_theta = self.psi[0]
						pre_phi = self.psi[1]

						self.__get_prob_loc_topic() ### 여기서 prob_loc_topic 갱신해 줘야함.
						# print('niter: %d\n' %niter, self.prob_loc_topic)
						# print('go EM')
						topic_post_prob = E(self.psi, self.prob_loc_topic, self.df_user, self.visited_loc_user)

						### ### 
						new_psi = M(self.visited_loc_user, topic_post_prob, self.psi, self.distance, self.N, self.Z, self.I, self.loc_Id)
						# new_psi = M2(self.prob_loc_topic, self.visited_loc_user, topic_post_prob, self.psi, self.distance, self.N, self.Z, self.I, self.loc_Id)
						##### 문제 0: Phi 갱신 값이 이상함. --> maybe fixed, 위에 수정한 부분 체크

						theta = new_psi[0]
						phi = new_psi[1].reshape(self.Z, self.I)
						self.psi = [theta, phi]

						test_condition = np.sqrt(np.sum(np.abs(pre_theta - theta)) + np.sum(np.abs(pre_phi - phi)))
						print(test_condition)

						condition = np.all((pre_theta - theta) < 1e-10) and np.all((pre_phi - phi) < 1e-10)
						
						if condition == True:
								break

				print("Finish parameter_estimation")
				
				return self.beta, self.psi

########################################################################################
		def get_location(self, current_location):
				'''
				This function changes the current location to the latitude and longitude
				For example, the current_location is "대전시 서구 복수동 475"
				This function returns the list containing latitude and longitude of that address
				'''
				current_address = parse.quote(str(current_location))
				address = urllib.request.urlopen("http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address=" + current_address).read().decode('utf-8')

				data = json.loads(address)
				latitude = data["results"][0]["geometry"]["location"]["lat"]
				longitude = data["results"][0]["geometry"]["location"]["lng"]
				return [latitude, longitude]


		def test(self, current_coordinate, psi, beta):
				'''
				This function calculates the probability of visiting location reflecting user's interest.
				In this step, we assume that all users are in the same space (in current address)
				If you are interested in the specific user, check the index of that user, then you can get the information of that user
				
				Input:
				1. L: the location latitude and longitude in the dataset
				2. current_coordinate: current address coordinates
				3. psi: estimated parameters
				4. beta: the activity area
				Output:
				The probabilities of location i per each user
				'''
				# pdb.set_trace()
				theta = psi[0]; phi = psi[1]
				current_distance = []
				for loc in self.L: # self.L 모든 location의 geo정보, np.array
						# pdb.set_trace()
						temp = np.exp(-0.5 * beta * np.inner(loc-current_coordinate, loc-current_coordinate))
						# temp = np.exp(-0.5 * beta * np.linalg.norm(loc - current_coordinate))
						current_distance.append(temp)

				# pdb.set_trace()
				current_distance = np.array(current_distance).reshape(1, -1)
				recommend_prob_numer = np.exp(phi) * current_distance # Z * I
				recommend_prob_denom = recommend_prob_numer.sum(axis=1).reshape(-1, 1)
				recommend_prob = recommend_prob_numer / recommend_prob_denom

				# pdb.set_trace()
				recommend_prob = theta @ recommend_prob # N * I
												  
				return recommend_prob

		def find_recommendation_random(self, num):
				# 이 함수에서는, N명에게 다 추천을 할 필요가 없음.
				# 어차피 밖에서 N명에 대해 for문을 돌고 있으므로
				# recommendation = []

				# for i in range(0, self.N): # N명에게
					# Randomly sample 100% of df['Restaurant Name']
				df_rand = self.df['Restaurant Name'].sample(frac=1.0)
					
					# Randomly sample n elements from df_rand (n = num)
				df_rand = df_rand.sample(n = num)

					# recommendation.append(df_rand.values.tolist())

				return df_rand.values.tolist() #recommendation

		def find_recommendation_max(self, num, idx_list): # all user have same recommendation
				df = self.df
				recommendation = []
				new_log_cnt = []
				for r_id in idx_list:
					try:
						df_restaurant_name = df.loc[df['Restaurant ID'] == r_id, 'Restaurant Name'].iloc[0]
						new_log_cnt.append(df_restaurant_name)
					except IndexError:
						pass

				for i in range(0, self.N):
					top_log = new_log_cnt[:num]
					recommendation.append(top_log)
				# for i in range(0, self.N):
				#     log_cnt = self.df['Restaurant Name'].value_counts()
				#     log_cnt = log_cnt.index.tolist()

				#     # Take the first n items from log_cnt(n = num)
				#     top_log = log_cnt[:num]
				#     recommendation.append(top_log)

				return recommendation

		def find_recommendation(self, recommend_prob):
				# pdb.set_trace()
				best_loc_id = np.argmax(recommend_prob, axis=1)
				best_loc = [self.loc_Id[x] for x in best_loc_id]
				recommendation = []
				for loc in best_loc:
						rest = self.df[self.df['Restaurant ID']==loc]['Restaurant Name'].unique()
						recommendation.append(rest[-1])
																  
				return recommendation

		def find_recommendation2(self, recommend_prob, num):
				# pdb.set_trace()
				best_loc_id = np.fliplr(recommend_prob.argsort()) # 확률이 큰 순서대로 
				best_loc_id = best_loc_id[:,:num]

				best_loc = [ [self.loc_Id[xx] for xx in x] for x in best_loc_id]
				
				recommendation = []
				for loc_num in best_loc:
						temp = []
						for loc in loc_num:
								rest = self.df[self.df['Restaurant ID']==loc]['Restaurant Name'].unique()
								temp.append(rest[-1])
						recommendation.append(temp)

				return recommendation

		def pre_at_N(self, recommend, test_data):
			pre_at_N = []
			for i in range(self.I):
				# pdb.set_trace()
				recommend_at_N = recommend[:,:(i+1)] # N * (i+1)

				accuracy = 0
				for n in range(self.N):
					if test_data[n] in recommend_at_N[n]:
						accuracy += 1
				accuracy = accuracy/self.N*100

				pre_at_N.append(accuracy)	
			
			return pre_at_N
		
		def MRR(self, recommend, test_data): # recommend는 numpy
			MRR = 0
			for n in range(self.N):
				t_recommend = recommend.tolist()
				mrr_index = t_recommend[n].index(test_data[n])+1
				MRR += 1/mrr_index
				MRR = MRR/self.N

			return MRR

		def topic_extraction(self):
			df_code = np.load('df_code.npy')
			df_subcode = np.load('df_subcode.npy')

			df_code = pd.DataFrame(df_code, columns=['code','name'])
			df_subcode = pd.DataFrame(df_subcode, columns=['subcode','name'])

			##
			df = self.df
			psi = self.psi

			df = df[['Restaurant ID','Restaurant Name','Restaurant code','Restaurant subcode']]

			df = df.sort_values('Restaurant ID')

			phi = psi[1]
			loc_topic = np.argmax(phi, axis=0) # 각 location별 가장 많이 가지고 있는 topic
			rest_id = sorted(df['Restaurant ID'].unique())

			# 위에서 구한 loc_topic에서 각 topic별로 rest_id를 모음 --> topic_info_resId
			topic_info_resId = [[] for i in range(phi.shape[0])]
			for i in range(phi.shape[1]): # I
				topic_info_resId[loc_topic[i]].append(rest_id[i])

			# pdb.set_trace()
			# 위에서 구한 topic_info_resId의 Restaurant code를 매칭
			topic_info = [[] for i in range(phi.shape[0])]
			topic_info_sub = [[] for i in range(phi.shape[0])]
			for i in range(phi.shape[0]): # Z
				rest_idx = topic_info_resId[i]
				
				for rest_idxx in rest_idx:
					df_idx = df[df['Restaurant ID']==rest_idxx].index.tolist()
					df_idx = df_idx[-1]
					code = df.loc[df_idx, ['Restaurant code']].tolist()
					subcode = df.loc[df_idx, ['Restaurant subcode']].tolist()
					topic_info[i] += df_code[df_code['code']==str(code[0])]['name'].tolist()
					topic_info_sub[i] += df_subcode[df_subcode['subcode']==str(subcode[0])]['name'].tolist()
					# topic_info[i] += df.loc[df_idx, ['Restaurant code']].tolist()
					# topic_info[i].append(df.loc[df_idx, ['Restaurant code']])
			
			return topic_info, topic_info_sub
			