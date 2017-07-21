import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import pdb


def load_data(filename):
    df = pd.read_csv(filename, delimiter='\t', index_col=False)
    return df

def initialize(N, Z, I):
    '''
    Initialize theta and phi
    theta shape: N * Z
    phi shape: Z * I
    '''
    theta = np.random.rand(N, Z)
    col_sum = theta.sum(axis=0)
    theta = theta / col_sum
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

        df_temp[usr_vsted_loc] = prob
        pXum[key] = df_temp.fillna(0)
        
    return pXum


def E(psi, pXum, df_user, x_um):
    np.seterr(divide='ignore', invalid='ignore')
    theta = psi[0]; phi = psi[1]; topicProb = {}
    memId = df_user['Member ID'].unique()
    
    theta = pd.DataFrame(theta, index=memId)
    for key in x_um.keys():
        theta_usr = theta.loc[key].as_matrix()
        pXum_usr = pXum[key].as_matrix()
        theta_usr = theta_usr.reshape(1, -1)
        temp = theta_usr.T * pXum_usr
        tempSum = temp.sum(axis=0)
        prob = temp / tempSum
        topicProb[key] = prob
            
    return topicProb
    

def get_ind(x_um):
    indices = []
    for key in x_um.keys():
        value = x_um[key]
        if len(value) == 1:
            indices.append([key, value[0]])
        else:
            for loc in value:
                indices.append([key, loc])

    return indices


def theta_optimize(pXum):
    theta_hat_numer = pXum.sum(axis=1)
    theta_hat_denom = theta_hat_numer.sum()
    theta_hat = theta_hat_numer / theta_hat_denom

    return theta_hat

def fun1(phi, theta, P_hat, dist):

    feed_dict = {P_hat : P_hat,
                 Theta : theta,
                 Phi : phi,
                 dist : dist}
    
    Q_ = sess.run(Q, feed_dict)

    Phi_grad_ = sess.run(Phi_grad, feed_dict)

    return -(Q_), Phi_grad_

def M(x_um,P_hat, Psi, dist, pXum):
    theta = []
    for key in P_hat.keys():
        temp = theta_optimize(P_hat[key])
        theta.append(temp)

    theta = np.array(theta)
    phi = Psi[1]
    P = pXum

    N = theta.shape[0]; Z = theta.shape[1]; I = phi.shape[1]

    #
    indices = get_ind(x_um)
    Indices = tf.SparseTensor(indices = indices, values = tf.ones(len(indices), dtype = tf.float64), dense_shape = [N, I])

    # placeholder
    P_hat = tf.placeholder(tf.float64, shape = [Z, N, I])
    Theta = tf.placeholder(tf.float64, shape = [N, Z])
    Phi = tf.placeholder(tf.float64, shape = [Z, I])
    dist = tf.placeholder(tf.float64, shape = [I, I])

    log_Theta = tf.expand_dims(tf.transpose(tf.log(Theta)), axis = 2)
    
    loglike = P_hat * log_Theta * P

    Q = tf.reduce_sum(tf.sparse_tensor_dense_matmul(Indices, tf.transpose(tf.reshape(loglike, [-1, I]))))

    Phi_grad = tf.gradients(Q, Phi)

    fun = lambda phi : fun1(phi, theta, P_hat, dist)
    
    res =  minimize(fun, phi, jac = True)

    return [theta, res.x]


def main():
    filename = "../data/daejeon.csv"
    df = load_data(filename)
    beta = 1
    Z = 5
    N = len(df['Member ID'].unique())
    I = len(df['Restaurant ID'].unique())

    df_loc = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_user = df[['Member ID', 'Restaurant ID']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))

    df_dist = getDist(beta, location)

    x_um = get_Xum(df)
    
    Psi = initialize(N, Z, I)
    Theta = Psi[0]; Phi = Psi[1]


    pXum = get_P_Xum(location, df_dist, x_um, Psi)    
    #pdb.set_trace()

    print("=====================================================")
    print("In the E step")
    print("=====================================================")
    
    P_hat = E(Psi, pXum, df_user, x_um)
    print("=====================================================")
    print('In the M step')
    print("=====================================================")
    # P :  pXum???
    res = M(x_um, P_hat, Psi, df_dist, pXum)
    phi = res[1]

#    print(phi[59])

    return res

if __name__=="__main__":
    main()
