import pandas as pd
import numpy as np

'''
Auther: Sumin Lim (KAIST)

This program implements the paper - Yang D. et al. "A Sentiment-Enhanced Personalized Location Recommendation System"

'''
def load_data():
    '''
    load data. This file uses temp data (daejeon.csv)
    '''
    df = pd.read_csv('/data/daejeon.csv', delimiter='\t', index_col=False)
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
    pref_final = pref_checkin - np.sign(pref_checkin - pref_sentiment) * np.
    return pref_final

def main():
    
    load_data()
    
    pref_checkin,pref_sentiment = get_pref_mats()
    # input: 
    # output: ndarray
    
    pref_final = compute_pref_final(pref_checkin, pref_sentiment) # R
    # input:
    # output: ndarray
    # Eq. (1)
    
    N, I = pref_final.shape
    # N: # of users
    # I: # of locations
    
    Z = int(input()) 
    # Z: user-latent space, location-latent space
    
    U,V = get_UV(N,I,Z)
    # random initialization
    
    while until converge:
        simU = get_similarity(U)
        simV = get_similarity(V)
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
    
