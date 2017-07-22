import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform # distance.euclidean(a,b)
import tensorflow as tf

def load_data():
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
    input
    beta: scalar
    location: should be sorted list
    
    output
    distance: dataframe, shape of I * I
    '''
    I = len(location)
    L = np.array([[x[1], x[2]] for x in location])
    dist = squareform(pdist(np.exp(-0.5*beta*L)))
    loc_id = np.array([x[0] for x in location])
    distance = pd.DataFrame(dist, columns=loc_id, index=loc_id)
    distance[distance == 0] = 1
    return distance

def get_Xum(df):
    df_temp = df.sort_values('Member ID')
    df_user = df_temp[["Member ID", "Restaurant ID"]]

    x_um = {}
    for index, row in df_user.iterrows():
        if row["Member ID"] not in x_um:
            x_um[row["Member ID"]] = [row["Restaurant ID"]]
        else:
            x_um[row["Member ID"]].append(row["Restaurant ID"])

    return x_um

def get_P_Xum(location, df_dist, x_um, psi):

    loc_id = np.array([x[0] for x in location])
    phi = psi[1]; phi = np.exp(phi)
    phi = pd.DataFrame(phi, columns=loc_id)
    pXum = {}; I = len(loc_id); Z = phi.shape[0]
    for key in x_um.keys():
        temp = np.full([Z, I], np.nan)
        df_temp = pd.DataFrame(temp, columns=loc_id)
        usr_vsted_loc = x_um[key]
        userdist = df_dist.loc[usr_vsted_loc, usr_vsted_loc]
        userdistSum = userdist.sum(axis=1)
        usr_phi = phi.loc[:, usr_vsted_loc]
        prob = usr_phi * userdistSum
        prob = prob / prob.sum()

        df_temp[usr_vsted_loc] = prob
        pXum[key] = df_temp.fillna(0)

    return pXum

def E(psi, pXum, df_user, x_um):
    np.seterr(divide='ignore', invalid='ignore')
    theta = psi[0]; phi = psi[1]; topicProb = {}
    memId = sorted(df_user['Member ID'].unique())
    locId = sorted(df_user['Restaurant ID'].unique())
    
    theta = pd.DataFrame(theta, index=memId)
    for key in x_um.keys():
        theta_usr = theta.loc[key].as_matrix()
        pXum_usr = pXum[key].as_matrix()
        theta_usr = theta_usr.reshape(1, -1)
        temp = theta_usr.T * pXum_usr
        tempSum = temp.sum(axis=0)
        prob = temp / tempSum
        topicProb[key] = pd.DataFrame(prob, columns=locId).fillna(0)
    
    return topicProb

def get_ind(x_um, locId):
    indices = []
    memIdx = -1
    for key in x_um.keys():
        memIdx += 1
        locs = x_um[key]
        for loc in locs:
            temp_loc = locId.index(loc)
            indices.append([memIdx, temp_loc])

    return indices


def theta_optimize(pXum):
    theta_hat_numer = pXum.sum(axis=1)
    theta_hat_denom = theta_hat_numer.sum()
    theta_hat = theta_hat_numer / theta_hat_denom
    return theta_hat

def M(x_um, p_hat, Psi, distance, N, Z ,I, locId):
    theta = []
    for key in p_hat.keys():
        temp = theta_optimize(p_hat[key])
        theta.append(temp)

    theta = np.array(theta)
    phi = Psi[1]
    phat = []

    for key in p_hat.keys():
        phat.append(p_hat[key].as_matrix())


    phat = np.array(phat)
    phat1 = np.swapaxes(phat, 0, 1)

    indices = get_ind(x_um, locId)
    Indices = tf.SparseTensor(indices = indices, values = tf.ones(len(indices), dtype = tf.float64), dense_shape = [N, I])

    # placeholder
    P_hat = tf.placeholder(tf.float64, shape = [Z, N, I])
    Theta = tf.placeholder(tf.float64, shape = [N, Z])
    Phi = tf.placeholder(tf.float64, shape = [Z, I])
    Dist = tf.placeholder(tf.float64, shape = [I, I])
    
    # Calculate P(x_um|z, R_u, Psi)
    # Z * I
    front = tf.exp(Phi)

    # N x I
    back = tf.sparse_tensor_dense_matmul(Indices, Dist)

    P_numer = tf.expand_dims(front, axis =1) * back
    P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis = 2), axis = 2)
    P = P_numer / P_denom

    log_Theta = tf.expand_dims(tf.transpose(tf.log(Theta)), axis = 2)

    loglike = P_hat * log_Theta * P

    Q = -tf.reduce_sum(tf.sparse_tensor_dense_matmul(Indices, tf.transpose(tf.reshape(loglike, [-1, I]))))
    Phi_grad = tf.negative(tf.gradients(Q, Phi))

    sess = tf.Session()


    def objective(phi_):
        phi_ = phi_.reshape(Z, I)
        #phat_, theta_, phi_, distance_ = param
        feed_dict={P_hat: phat1, Theta: theta, Phi: phi_, Dist: distance}
        return sess.run(Q, feed_dict={P_hat: phat1, Theta: theta, Phi: phi_, Dist: distance})

    def gradient(phi_):
        phi_ = phi_.reshape(Z, I)
        #phat_, theta_, phi_, distance_ = param
        feed_dict={P_hat: phat1, Theta: theta, Phi: phi_, Dist: distance}
        ret = sess.run(Phi_grad, feed_dict={P_hat: phat1, Theta: theta, Phi: phi_, Dist: distance})
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

    locId = sorted(df_user['Restaurant ID'].unique())
    memId = sorted(df_user['Member ID'].unique())
    df_dist = getDist(beta, location)

    x_um = get_Xum(df)
    

    '''
    Algorithm 2 Parameter initialization
    '''
    Psi = initialize(N, Z, I)
    
    pXum = get_P_Xum(location, df_dist, x_um, Psi)    
    distance = df_dist.as_matrix()

    while True:

        pre_theta = Psi[0]
        pre_phi = Psi[1]
        
        P_hat = E(Psi, pXum, df_user, x_um)

        psi = M(x_um, P_hat, Psi, distance, N, Z, I, locId)
        
        theta = psi[0]
        phi = psi[1].reshape(Z, I)
        
        # update Psi
        Psi = [theta, phi]

        if (np.all(pre_theta - theta) < 1e-6) and (np.all(pre_phi - phi) < 1e-6):
            break


    return Psi
    # # test ::  beta value: 1 , topic: 5 
    # t_theta = Psi[0]; t_phi = Psi[1].reshape(5, 852)
    # print(t_theta.shape) #(1153, 5)
    # print(t_phi.shape) #(5, 852)

if __name__ == '__main__':
    main()