'''
Author: Sumin Lim (KAIST), Hyunji Lee (Jeonbuk Univ.)
July 23th. 2017

Paper: Kurashima T. et al., Geo Topic Model: Joint Modeling of User's Acitivity Area and Interests for Location Recommendation

This program implements location recommendation using geotag data
'''

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
	for key in visited_loc_user.keys():
		theta_usr = theta.loc[key].as_matrix()
		prob_loc_topic_usr = prob_loc_topic[key].as_matrix()
		theta_usr = theta_usr.reshape(1, -1)
		temp = theta_usr.T * prob_loc_topic_usr
		temp_sum = temp.sum(axis=0)
		prob = temp / temp_sum
		topic_prob[key] = pd.DataFrame(prob, columns=loc_Id).fillna(0)
	
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

def M(visited_loc_user, topic_posterior_prob, Psi, distance, N, Z ,I, loc_Id):
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
	theta = []
	for key in topic_posterior_prob.keys():
		temp = theta_optimize(topic_posterior_prob[key])
		theta.append(temp)

	theta = np.array(theta)
	phi = Psi[1]
	phat = []

	for key in topic_posterior_prob.keys():
		phat.append(topic_posterior_prob[key].as_matrix())


	phat = np.array(phat)
	phat1 = np.swapaxes(phat, 0, 1)

	indices = get_ind(visited_loc_user, loc_Id)
	Indices = tf.SparseTensor(indices = indices, values = tf.ones(len(indices), dtype = tf.float64), dense_shape = [N, I])

	# placeholder
	Topic_post_prob = tf.placeholder(tf.float64, shape = [Z, N, I])
	Theta = tf.placeholder(tf.float64, shape = [N, Z])
	Phi = tf.placeholder(tf.float64, shape = [Z, I])
	Dist = tf.placeholder(tf.float64, shape = [I, I])
	
	# Calculate P(visited_loc_user|z, R_u, Psi)
	# Z * I
	front = tf.exp(Phi)

	# N x I
	back = tf.sparse_tensor_dense_matmul(Indices, Dist)

	P_numer = tf.expand_dims(front, axis =1) * back
	P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis = 2), axis = 2)
	P = P_numer / P_denom

	log_Theta = tf.expand_dims(tf.transpose(tf.log(Theta)), axis = 2)

	loglike = Topic_post_prob * log_Theta * P

	Q = -tf.reduce_sum(tf.sparse_tensor_dense_matmul(Indices, tf.transpose(tf.reshape(loglike, [-1, I]))))
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

	def __init__(self, df, beta, Z, N, I):
		
		self.df = df
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
		self.__get_visited_loc_user()

		# out: self.psi
		self.__psi_initialize()

		# out: self.prob_loc_topic
		self.__get_prob_loc_topic()  

	def __getDist(self):
		'''
		Calculate the distance between locations 

		Input
		beta: scalar
		location: should be sorted list
		
		Output
		distance: dataframe, shape of I * I, row and column index = restaurant ID
		'''
		I = len(self.location)
		L = np.array([[x[1], x[2]] for x in self.location])
		dist = squareform(pdist(np.exp(-0.5*self.beta*L)))
		loc_Id = np.array([x[0] for x in self.location])
		distance = pd.DataFrame(dist, columns=loc_Id, index=loc_Id)
		distance[distance == 0] = 1
		
		self.df_dist = distance



	def __get_visited_loc_user(self):
		'''
		Get the visited locations per user

		Input: DataFrame

		Output: Dictionary, key = Member ID, value = [restaurant ID, ... , restaurant ID]
		The length of each value differs from each user (len(value) = Mu)
		'''
		df_temp = self.df.sort_values('Member ID')
		df_user = df_temp[["Member ID", "Restaurant ID"]]
		
		# visited_loc_user : x_um
		visited_loc_user = {}
		for index, row in df_user.iterrows():
			if row["Member ID"] not in visited_loc_user:
				visited_loc_user[row["Member ID"]] = [row["Restaurant ID"]]
			else:
				visited_loc_user[row["Member ID"]].append(row["Restaurant ID"])

		self.visited_loc_user = visited_loc_user


	def __psi_initialize(self):
		'''
		Initialize theta and phi
		theta shape: N * Z
		phi shape: Z * I
		'''
		theta = np.random.rand(self.N, self.Z)
		row_sum = theta.sum(axis=1).reshape(-1, 1)
		theta = theta / row_sum
		phi = np.random.rand(self.Z, self.I)
		
		self.psi = [theta, phi]

	def __get_prob_loc_topic(self):
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
		prob_loc_topic = {}; I = len(loc_Id); Z = phi.shape[0]

		for key in self.visited_loc_user.keys():
			temp = np.full([Z, I], np.nan)
			df_temp = pd.DataFrame(temp, columns=loc_Id)
			usr_vsted_loc = self.visited_loc_user[key]
			user_dist = self.df_dist.loc[usr_vsted_loc, usr_vsted_loc]
			uuser_dist_sum = user_dist.sum(axis=1)
			usr_phi = phi.loc[:, usr_vsted_loc]
			prob = usr_phi * uuser_dist_sum
			prob = prob / prob.sum()

			df_temp[usr_vsted_loc] = prob
			prob_loc_topic[key] = df_temp.fillna(0)

		self.prob_loc_topic = prob_loc_topic


	def trainParams(self, maxiter):

		self.__initialize()

		niter = 0
		cnt = 0
		
		for niter in range(1, maxiter):
			cnt += 1
			print("==================== In the While Loop =======================")
			print(" %d th iteration" % cnt)

			pre_theta = self.psi[0]
			pre_phi = self.psi[1]

			topic_post_prob = E(self.psi, self.prob_loc_topic, self.df_user, self.visited_loc_user)

			new_psi = M(self.visited_loc_user, topic_post_prob, self.psi, self.distance, self.N, self.Z, self.I, self.loc_Id)

			theta = new_psi[0]
			phi = new_psi[1].reshape(self.Z, self.I)
			self.psi = [theta, phi]

			condition = np.all((pre_theta - theta) < 1e-10) and np.all((pre_phi - phi) < 1e-10)
			
			if condition == True:
				break

		print("Finish parameter_estimation")
		
		return self.beta, self.psi
		

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
		theta = psi[0]; phi = psi[1]
		current_distance = []
		for loc in self.L:
			temp = np.exp(-0.5 * beta * np.linalg.norm(loc - current_coordinate))
			current_distance.append(temp)

		current_distance = np.array(current_distance).reshape(1, -1)
		recommend_prob_numer = phi * current_distance
		recommend_prob_denom = recommend_prob_numer.sum(axis=1).reshape(-1, 1)
		recommend_prob = recommend_prob_numer / recommend_prob_denom

		recommend_prob = theta @ recommend_prob
						  
		return recommend_prob

	def find_recommendation(self, recommend_prob):
		best_loc_id = np.argmax(recommend_prob, axis=1)
		best_loc = [self.loc_Id[x] for x in best_loc_id]
		recommendation = []
		for loc in best_loc:
			rest = self.df[self.df['Restaurant ID']==loc]['Restaurant Name'].unique()
			recommendation.append(rest[-1])
								  
		return recommendation