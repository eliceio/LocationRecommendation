import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform # distance.euclidean(a,b)
import tensorflow as tf

def load_data():
    df = pd.read_csv('../data/daejeon.csv', delimiter='\t', index_col=False)
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
    input
    beta: scalar
    location: should be sorted list
    
    output
    distance: dataframe, shape of I * I
    '''
    I = len(location)
    L = np.array([[x[1], x[2]] for x in location])
    dist = squareform(pdist(np.exp(-0.5*beta*L)))
    loc_Id = np.array([x[0] for x in location])
    distance = pd.DataFrame(dist, columns=loc_Id, index=loc_Id)
    distance[distance == 0] = 1
    return distance

def get_visited_loc_user(df):
    df_temp = df.sort_values('Member ID')
    df_user = df_temp[["Member ID", "Restaurant ID"]]

    visited_loc_user = {}
    for index, row in df_user.iterrows():
        if row["Member ID"] not in visited_loc_user:
            visited_loc_user[row["Member ID"]] = [row["Restaurant ID"]]
        else:
            visited_loc_user[row["Member ID"]].append(row["Restaurant ID"])

    return visited_loc_user

def get_prob_loc_topic(location, df_dist, visited_loc_user, psi):

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
    theta_hat_numer = prob_loc_topic.sum(axis=1)
    theta_hat_denom = theta_hat_numer.sum()
    theta_hat = theta_hat_numer / theta_hat_denom
    return theta_hat

def M(visited_loc_user, topic_posterior_prob, Psi, distance, N, Z ,I, loc_Id):
    '''
    topic_posterior_prob: topic_posteior_probability
    Psi: [theta, phi]

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

    loglike = topic_post_prob * log_Theta * P

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

def main():
    df = load_data()
    beta = float(input("Enter the beta value:"))
    Z = int(input("Enter the number of topic:"))
    N = len(df['Member ID'].unique())
    I = len(df['Restaurant ID'].unique())

    df_loc = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_user = df[['Member ID', 'Restaurant ID']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))


    loc_Id = sorted(df_user['Restaurant ID'].unique())
    #mem_Id = sorted(df_user['Member ID'].unique())
    df_dist = getDist(beta, location)
    distance = df_dist.as_matrix()

    # visited_loc_user : visited location of user 
    # visited_loc_user : before x_um
    visited_loc_user = get_visited_loc_user(df)
    
    
    # Algorithm 2 : Parameter initialization
    # psi : [theta, phi]
    # theta: probability that topic Z is chosen by user N
    # phi: probability that location I is chosen for topic Z
    
    psi = initialize(N, Z, I)

    # prob_loc_topic : the probability that each location is chosen from topic Z (before name: pXum)
    prob_loc_topic = get_prob_loc_topic(location, df_dist, visited_loc_user, psi)    
   

    while True:
        pre_theta = Psi[0]
        pre_phi = Psi[1]
        
        # topic_post_prob : topic posterior probability
        # topic_post_prob : before P_hat
        topic_post_prob = E(psi, prob_loc_topic, df_user, visited_loc_user)

        new_psi = M(visited_loc_user, topic_post_prob, psi, distance, N, Z, I, loc_Id)
        
        theta = new_psi[0]
        phi = new_psi[1].reshape(Z, I)
        
        # update Psi
        psi = [theta, phi]

        if (np.all(pre_theta - theta) < 1e-6) and (np.all(pre_phi - phi) < 1e-6):
            break

    return psi


if __name__ == '__main__':
    main()
