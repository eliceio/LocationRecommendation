import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import pearsonr, logistic
from scipy.special import expit
from scipy.optimize import minimize

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

    sim_u = np.array(sim_u)
    return sim_u


def get_sim_v(df):
    '''
    For two venues, the similarity score is set to 1 if both venues have the same sub-category in Foursquare
    and set 0 if there is no overlapping sub-category

    If two restaurants have the same cuisine code, similarity score is set to 1, else 0.
    '''
    location = []
    for index, row in df.iterrows():
        tmp = [row['Restaurant ID'], row['Restaurant code']]
        if tmp not in location:
            location.append(temp)

    sim_v = []
    I = len(location)
    
    for i in range(I):
        temp = []
        current_code = location[i][1]
        for j in range(I):
            if location[j][1] == current_code:
                temp.append(1)
            else:
                temp.append(0)
        sim_v.append(temp)
        
    return sim_v


def get_coefficient(pref_final, sim_u, sim_v, U, V):
    '''
    lambda_u = sigma^2_R / sigma^2_U
    lambda_v = sigma^2_R / sigma^2_V
    alpha = sigma^2_R / sigma^2_simU
    beta = sigma^2_R / sigma^2_simV
    '''
    var_R = np.var(pref_final)
    lambda_u = var_R / np.var(U)
    lambda_v = var_R / np.var(V)
    alpha = var_R / np.var(sim_u)
    beta = var_R / np.var(sim_v)
    
    return lambda_u, lambda_v, alpha, beta


def get_log_posterior(U, V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z):
    '''
    Calculate the log posterior probability of U and V keeping the variance parameter fixed. 
    In later, minimize the log posterior which is the return value of this function
    '''
    U = sp.resize(U, (N, Z))
    V = sp.resize(V, (Z, I))
    first_term = np.sum(pref_final - expit(U @ V))
    second_term = lambda_u * np.sum(U @ U.T) + lambda_v * np.sum(V @ V.T)
    third_term = alpha * np.sum((U - (sim_u @ U)) @ (U - (sim_u @ U)).T)
    fourth_term = beta * np.sum((V.T - (sim_v @ V.T)) @ (V.T - (sim_v @ V.T)).T)
    log_posterior = 0.5 * (first_term + second_term + third_term + fourth_term)
    
    return log_posterior


def get_grad_u(U, V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z):
    
    U = sp.resize(U, (N, Z))
    V = sp.resize(V, (Z, I))
    
    grad_u_first = (logistic.pdf(U @ V) * (expit(U @ V) - pref_final)) @ V.T
    grad_u_second = lambda_u * U + alpha * (U - sim_u @ U)
    grad_u_third = -alpha * (sim_u @ (U - sim_u @ U))
    grad_u = grad_u_first + grad_u_second + grad_u_third

    grad_u = np.ndarray.flatten(grad_u)

    return grad_u


def get_grad_v(U, V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z):

    U = sp.resize(U, (N, Z))
    V = sp.resize(V, (Z, I))
    
    grad_v_first = (logistic.pdf(U @ V) * (expit(U @ V)-pref_final)).T @ U
    grad_v_second = (lambda_v * V).T + beta * (V.T - sim_v @ V.T)
    grad_v_third = -beta * (sim_v @ (V.T - sim_v @ V.T))
    grad_v = grad_v_first + grad_v_second + grad_v_third

    grad_v = np.ndarray.flatten(grad_v)

    return grad_v


def compute_metrics(U, V, pref_final):
    R_hat = U @ V
    T = pref_final.shape[0] * pref_final.shape[1]
    MAE = np.sum(np.abs(pref_final - R_hat)) / T
    RMSE = np.sqrt(np.sum(np.square(pref_final - R_hat)) / T)
    return MAE, RMSE

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

    # get similarity of venues from dataset
    sim_v = get_sim_v(df)
    
    # while until converge:
        # get similarity of users with pearson correlation coefficient
    sim_u = get_sim_u(U)
        
    lambda_u, lambda_v, alpha, beta = get_coefficient(pref_final, sim_u, sim_v, U, V)
    
    log_posterior = get_log_posterior(U, V, pref_final, simU, simV, lambda_u, lambda_v, alpha, beta, N, I, Z)
        # in particular, objective function
        # Eq. (14)

    while True:
        u_res = minimize(get_log_posterior,
                         x0 = U, args = (V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z),
                         jac = get_grad_u)

        v_res = minimize(get_log_posterior,
                         x0 = V, args = (U, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z),
                         jac = get_grad_v)

        estimated_U = u_res.x.reshape(N, Z)
        estimated_V = v_res.x.reshape(Z, I)

        cond = np.sqrt(np.sum(np.square(U - estimated_U)) + np.sum(np.square(V - estimated_V)))
        condition = cond < 1e-06

        print("condition value:", cond)
        print("U:", U)
        print("V:", V)
        print("estimated_U:", estimated_U)
        print("estimated_V:", estimated_V)
        print("condition:", condition)

        if condition:
            break

        U, V = estimated_U, estimated_V

        lambda_u, lambda_v, alpha, beta = get_coefficient(pref_final, sim_u, sim_v, U, V)


    MAE, RMSE = compute_metrics(U, V, pref_final)
    print("MAE:", MAE)
    print("RMSE:", RMSE)
                                      

    ## 끝 U, V 
    
    # performance evaluation
    ## Eq. (17) & (18)
    

    # Recommendation
    
