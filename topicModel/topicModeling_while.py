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
import json

'''
Author: Sumin Lim (KAIST), Hyunji Lee (Jeonbook Univ.)
July 23th. 2017

Paper: Kurashima T. et al., Geo Topic Model: Joint Modeling of User's Acitivity Area and Interests for Location Recommendation

This program implements location recommendation using geotag data
'''

def load_data():
    '''
    Read data file
    File format: .csv, separated by tab
    '''
    df = pd.read_csv('daejeon.csv', delimiter='\t', index_col=False)
    return df

def initialize(N, Z, I):
    '''
    Initialize theta and phi
    theta shape: N * Z
    phi shape: Z * I
    '''
    theta = np.random.rand(N, Z)
    row_sum = theta.sum(axis=1).reshape(-1, 1)
    theta = theta / row_sum
    phi = np.random.rand(Z, I)
    
    return [theta, phi]

def getDist(beta, location):
    '''
    Calculate the distance between locations 

    Input
    beta: scalar
    location: should be sorted list
    
    Output
    distance: dataframe, shape of I * I, row and column index = restaurant ID
    '''
    I = len(location)
    L = np.array([[x[1], x[2]] for x in location])
    dist = squareform(pdist(np.exp(-0.5*beta*L)))
    loc_Id = np.array([x[0] for x in location])
    distance = pd.DataFrame(dist, columns=loc_Id, index=loc_Id)
    distance[distance == 0] = 1
    return distance

def get_visited_loc_user(df):
    '''
    Get the visited locations per user

    Input: DataFrame

    Output: Dictionary, key = Member ID, value = [restaurant ID, ... , restaurant ID]
    The length of each value differs from each user (len(value) = Mu)
    '''
    df_temp = df.sort_values('Member ID')
    df_user = df_temp[["Member ID", "Restaurant ID"]]
    
    # visited_loc_user : x_um
    visited_loc_user = {}
    for index, row in df_user.iterrows():
        if row["Member ID"] not in visited_loc_user:
            visited_loc_user[row["Member ID"]] = [row["Restaurant ID"]]
        else:
            visited_loc_user[row["Member ID"]].append(row["Restaurant ID"])

    return visited_loc_user

def get_prob_loc_topic(location, df_dist, visited_loc_user, psi):
    '''
    P(i|z, R_u, Phi), equation (2) in the paper
    location i is chosen from topic z after consideration of the user's geotags R_u

    Input:
    location (dataFrame), df_dist (dataFrame), visited_loc_user (dictionary), psi ([theta, phi])

    Output:
    Dictionary with key = Member ID, value = Z * I dataFrame
    '''
    loc_Id = np.array([x[0] for x in location])
    phi = psi[1]; phi = np.exp(phi)
    phi = pd.DataFrame(phi, columns=loc_Id)
    prob_loc_topic = {}; I = len(loc_Id); Z = phi.shape[0]

    for key in visited_loc_user.keys():
        temp = np.full([Z, I], np.nan)
        df_temp = pd.DataFrame(temp, columns=loc_Id)
        usr_vsted_loc = visited_loc_user[key]
        user_dist = df_dist.loc[usr_vsted_loc, usr_vsted_loc]
        uuser_dist_sum = user_dist.sum(axis=1)
        usr_phi = phi.loc[:, usr_vsted_loc]
        prob = usr_phi * uuser_dist_sum
        prob = prob / prob.sum()

        df_temp[usr_vsted_loc] = prob
        prob_loc_topic[key] = df_temp.fillna(0)

    return prob_loc_topic

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
    Phi_grad = tf.negative(tf.gradients(Q, Phi))

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

def cnt_visited_location(x_um):
    '''
    This function returns the dictionary with key = member ID, value: the number of re-visiting the location
    '''
    cnt_visited_loc_usr={}
    for key in x_um:
        usr = Counter(x_um[key])
        dict_usr = dict(usr)
        cnt_visited_loc = dict(Counter(dict_usr.values()))
        cnt_visited_loc_usr[key] = cnt_visited_loc

    return cnt_visited_loc_usr

def parameter_estimation():
    df = load_data()
    beta = float(input("Enter the beta value:"))
    Z = int(input("Enter the number of topic:"))
    N = len(df['Member ID'].unique())
    I = len(df['Restaurant ID'].unique())

    df_loc = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_user = df[['Member ID', 'Restaurant ID']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))


    loc_Id = sorted(df_user['Restaurant ID'].unique())
    mem_Id = sorted(df_user['Member ID'].unique())
    df_dist = getDist(beta, location)
    distance = df_dist.as_matrix()

    # visited_loc_user : visited location of user 
    # visited_loc_user : before x_um
    visited_loc_user = get_visited_loc_user(df)
    
    # cnt_visited_loc_usr: 유저당 가게 중복 방문 수 
    cnt_visited_loc_usr = cnt_visited_location(visited_loc_user)
    print(cnt_visited_loc_usr)
    
    
    # Algorithm 2 : Parameter initialization
    # psi : [theta, phi]
    # theta: probability that topic Z is chosen by user N
    # phi: probability that location I is chosen for topic Z
    psi = initialize(N, Z, I)
    
    # prob_loc_topic : the probability that each location is chosen from topic Z (before name: pXum)
    prob_loc_topic = get_prob_loc_topic(location, df_dist, visited_loc_user, psi)    
   
    # counting while loop
    cnt_loop = 0

    while True:

        pre_theta = psi[0]
        pre_phi = psi[1]
        
        # topic_post_prob : topic posterior probability
        # topic_post_prob : before P_hat
        topic_post_prob = E(psi, prob_loc_topic, df_user, visited_loc_user)

        new_psi = M(visited_loc_user, topic_post_prob, psi, distance, N, Z, I, loc_Id)
        
        # update Psi
        theta = new_psi[0]
        phi = new_psi[1].reshape(Z, I)
        psi = [theta, phi]

        if (np.all(pre_theta - theta) < 1e-6) and (np.all(pre_phi - phi) < 1e-6):
            break
        
        print("count loop: "cnt_loop)
        cnt_loop =+ 1

    return beta, psi


def get_location(current_location):
    '''
    This function changes the current location to the latitude and longitude
    For example, the current_location is "대전시 서구 복수동 475"
    This function returns the list containing latitude and longitude of that address
    '''
    current_address = parse.quote(current_location)
    address = urllib.request.urlopen("http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address=" + location).read()

    json = json.loads(address)
    latitude = json["results"][0]["geometry"]["location"]["lat"]
    longitude = json["results"][0]["geometry"]["location"]["lng"]
    return [latitude, longitude]


def test(L, current_coordinate, psi, beta):
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
    for loc in L:
        temp = np.exp(-0.5 * beta * np.linalg.norm(loc - current_coordinate))
        current_distance.append(temp)

    current_distance = np.array(current_distance).reshape(1, -1)
    recommend_prob_numer = phi * current_distance
    recommend_prob_denom = recommend_prob_numer.sum(axis=1).reshape(-1, 1)
    recommend_prob = recommend_prob_numer / recommend_prob_denom

    recommend_prob = theta @ recommend_prob
                      
    return recommend_prob


def find_recommendation(recommend_prob, locId, df):
    best_loc_id = np.argmax(recommend_prob, axis=1)
    best_loc = [locId[x] for x in best_loc_id]
    recommendation = []
    for loc in best_loc:
        rest = df[df['Restaurant ID']==loc]['Restaurant Name'].unique()
        recommendation.append(rest[-1])
                              
    return recommendation

def main():
    df = load_data()
    df_location = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_user = df[['Member ID', 'Restaurant ID']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))
    locId = sorted(df_user['Restaurant ID'].unique())
    L = np.array([[x[1], x[2]] for x in location])
    
    beta, psi = parameter_estimation()
    current_location = input("Enter the current space:")
    current_coordinate = get_location(current_location)

    recommend_prob = test(L, current_coordinate, psi, beta)
    print(recommend_prob)

    recommendation = find_recommendation(recommend_prob, locId, df)
    print(recommendation)
    
    
if __name__ == '__main__':
    main()
