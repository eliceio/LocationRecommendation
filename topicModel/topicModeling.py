import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import sys


def load_data():
    df = pd.read_csv('data/대전시.csv', sep='\t', index_col=False)
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
    pXum = []; I = len(loc_id); Z = phi.shape[0]
    for key in x_um.keys():
        temp = np.full([Z, I], np.nan)
        df_temp = pd.DataFrame(temp, columns=loc_id)
        usr_vsted_loc = x_um[key]
        userdist = df_dist.loc[usr_vsted_loc, usr_vsted_loc]
        userdistSum = userdist.sum(axis=1)
        usr_phi = phi.loc[:, usr_vsted_loc]
        prob = usr_phi * userdistSum

        df_temp[usr_vsted_loc] = prob
        pXum.append(df_temp)
        
    return pXum


def E(psi, pXum, df_user, x_um):
    theta = psi[0]; phi = psi[1]; topicProb = {}
    memId = df_user['Member ID'].unique()
    
    theta = pd.DataFrame(theta, index=mem_id)
    for key in x_um.keys():
        theta_usr = theta.loc[key].as_matrix()
        pXum_usr = pXum[key].as_matrix()
        theta_usr = theta_usr.reshape(1, -1)
        temp = theta_usr.T * pXum_usr
        tempSum = temp.sum(axis=0)
        prob = temp / tempSum
        topicProb[key] = prob
            
    return topicProb
    

def main():
    df = load_data()
    beta = float(input("Enter the beta value:"))
    Z = int(input("Enter the number of topic:"))
    N = len(df['Member ID'].unique())
    I = len(df['Restaurant ID'].unique())

    df_loc = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    df_user = df[['Member ID', 'Restaurant ID']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))

    df_dist = getDist(beta, location)

    x_um = get_Xum(df)

if __name__=="__main__":
    main()
