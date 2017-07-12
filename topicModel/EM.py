import pandas as pd
import numpy as np


'''
이 코드는 단순히 git에 폴더를 올리기 위함이니 각 맡은 부분은 한 파일로 푸쉬해주시면 감사하겠습니다 :)
(나중에 import 해서 합칠 수 있게 해주시면 감사하겠습니다!
'''

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
