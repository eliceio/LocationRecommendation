import pandas as pd
import numpy as np
from scipy.stats import pearsonr

'''
Auther: Sumin Lim (KAIST)

This program implements the paper - Yang D. et al. "A Sentiment-Enhanced Personalized Location Recommendation System"

'''
def load_data():
    '''
    load data. This file uses temp data (daejeon.csv)
    '''
    df = pd.read_csv('/data/Daejeon_dataset.csv', delimiter='\t', index_col=False)
    return df

def get_pref_mats(df):
    '''
    This function generates the check-in matrix and sentiment-preference matrix
    
    Input:
    df, dataFrame. Return from load_data function

    Output:
    1. pref_checkin, check-in preference matrix
    2. pref_sentiment, sentiment preference matrix
    '''
    mem_id = sorted(df['Member ID'].unique()); loc_id = sorted(df['Restaurant ID'].unique())
    pref_checkin = pd.DataFrame(0, index=mem_id, columns=loc_id)
    pref_sentiment = pd.DataFrame(0, index=mem_id, columns=loc_id)
    
    for index, row in df.iterrows():
        # make sentiment preference matrix
        member = row['Member ID']; restaurant = row['Restaurant ID']; rating = row['Rating']
        pref_sentiment.loc[member, restaurant] = rating
        checkin = pref_checkin.loc[member, restaurant]
        if checkin == 0:
            checkin = 1
        elif checkin > 0:
            checkin += 1

        if checkin >= 3:
            checkin = 3

        pref_checkin.loc[member, restaurant] = checkin

    pref_checkin = np.array(pref_checkin); pref_sentiment = np.array(pref_sentiment)
    
    return pref_checkin, pref_sentiment

def compute_pref_final(pref_checkin, pref_sentiment):
    '''
    This function calculates the final preference matrix. 
    Equation (1) from the paper
    
    Input:
    1. pref_checkin, check-in preference matrix
    2. pref_sentiment, sentiment preference matrix

    Output:
    pref_final, final preference matrix
    '''
    pref_final = pref_checkin - np.sign(pref_checkin - pref_sentiment) * np.heaviside(np.abs(pref_checkin - pref_sentiment)-2, 0.5)
    return pref_final


def get_UV(N, I, Z):
    '''
    This function initialize two matrices U and V

    Input:
    1. N: the number of users
    2. I: the number of locations (items)
    3. Z: the dimension of latent space 

    Output:
    randomly initialized U,V 
    '''
    U = np.random.rand(N, Z)
    V = np.random.rand(Z, I)

    return U, V

def get_sim_u(pref_final):
    '''
    This function returns similarity of users

    Input: pref_final, which is the return matrix of the compute_pref_final function
    Output: similarity of users, N * N matrix
    '''
    N, _ = pref_final.shape
    sim_u = []
    for n in range(N):
        temp = []
        for i in range(N):
            temp.append(pearsonr(pref_final[n], pref_final[i])[0])
        sim_u.append(temp)

    sim_u = np.array(sim_U)
    return sim_u


def get_sim_v():
    '''
    For two venues, the similarity score is set to 1 if both venues have the same sub-category in Foursquare
    and set 0 if there is no overlapping sub-category

    So, we need additional data - category and sub-category for the restaurants
    '''
    return sim_v


def main():
    
    df = load_data()

    # input: dataFrame
    # output: ndarray
    pref_checkin,pref_sentiment = get_pref_mats(df)

    # input: ndarray
    # output: ndarray
    # Eq. (1)
    pref_final = compute_pref_final(pref_checkin, pref_sentiment) # R

    # N: # of users
    # I: # of locations
    N, I = pref_final.shape

    # Z: user-latent space, location-latent space    
    Z = int(input()) 

    # random initialization    
    U,V = get_UV(N,I,Z)

    
    while until converge:
        sim_u = get_sim_u(U)
        sim_v = get_sim_v(V)
        # pearson correlation coefficient
        
        lambda_u, lambda_v, alpha, beta = get_coefficient(R, simU, simV, U, V)
    
        log_posterior = get_log_posterior(R, simU, simV, U, V)
        # in particular, objective function
        # Eq. (14)
    
        grad_u = get_gradient(U, V, R, lambda_u, alpha, simU)
        grad_v = get_gradient(U, V, R, lambda_v,  beta, simV)
        
        minization()
    
    ## 끝 U, V 
    
    # performance evaluation
    ## Eq. (17) & (18)
    

    # Recommendation
    
