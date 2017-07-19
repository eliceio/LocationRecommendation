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

def get_P_Xum(location, df_dist, x_um):
    
    loc_id = np.array([x[0] for x in location])
    phi = np.exp(phi)
    phi = pd.DataFrame(phi, columns=loc_id)
    Px_um = []
    for key in x_um.keys():
        userDist = df_dist.loc[key, key]
        userDistSum = userDist.sum(axis=1)

        user_phi = phi.loc[:, key]
        Px_um.append(user_phi * userDistSum)

    return Px_um
    

def main():
    df = load_data()
    beta = int(input("Enter the beta value:"))
    N = len(df['Member ID'].unique())
    I = len(df['Restaurant ID'].unique())

    df_loc = df[['Restaurant ID', 'Restaurant Latitude', 'Restaurant Longitude']]
    location = sorted(list(set([tuple(x) for x in df_loc.to_records(index=False)])))

    df_dist = getDist(beta, location)

    x_um = get_Xum(df)

if __name__=="__main__":
    Z = int(sys.argv[1])
    main()
