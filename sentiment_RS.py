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
    
