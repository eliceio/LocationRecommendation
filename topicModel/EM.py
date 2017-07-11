import pandas as pd
import numpy as np

# Algorithm 2
def initialize():
    theta = np.array()
    phi = np.array()

# Algorithm 1
def EM():
    initialize()
    iter = 1
    while iter < max_iter:
        E()
        M()
        
    return

# Algorithm 4
def M(psi, probs):

    # Optimize theta_uz

    # Optimize phi_z = [phi_z1, ... , phi_zI] with gradient descent
    
    return psi
    
if __name__=="__main__":
    data = pd.read_csv('./data/대전시.csv', sep='\t', index_col=False)
    
    # nUser refers to the N in the Geotopic paper (the number of users)
    nUser = data["Member ID"].unique()

    # nLoc refers to the I in the Geotopic paper (the number of locations)
    nLoc = data["Restaurant Address"].unique()

    # nTopic refers to the Z in the paper (the number of latent topics)
    nTopic = input()

    initTheta, initPhi = initialize(nUser, nTopic, nLoc)
    EM()
