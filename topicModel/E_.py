
# Algorithm 2
def initialize(nUser, nTopic, nLoc):

    theta = np.random.rand(nUser, nTopic)
    col_sum = theta.sum(axis=0)
    theta = theta / col_sum

    phi = np.random.rand(nTopic, nLoc)

    return [theta, phi]

########################################################
# Parameter Estimation

# 우리가 찾는 parameter는 a set of user location logs X가 주어졌을 때, 
# (log) likelihood를 최대화 하는 parameter set.

# 이러한 parameter set을 찾기 위해 EM algorithm을 사용

# 먼저 E-step: the conditional expectation of the complete-data log likelihood 계산
# 아래 1),2) 필요
# 1) P(z|u,m; psi): 
'''the topic posterior probability of the mth location of user u
                   given the current estimate.
    user u가 방문한 m번째 장소에 대해, 이 장소가 topic z를 가질 확률. 
    모든 z에 대해 값이 다 있어야 함. 
    --> 길이 N인 리스트 안에, 
    --> [ user 0:  [ [확률z개],[확률z개],[확률z개],...,[확률z개] ], 
          user 1:  [ [확률z개],[확률z개],         ...,[확률z개] ], 
          ... 
          user N-1:[ [확률z개],[확률z개], 각 user별 길이는 M_u   ] ]
'''

# 2) likelihood: 
''' 식 3: 이는 log data를 가지는 user u가 어떤 장소 i를 방문할 확률을 계산한 
식1에서, 장소가 log data의 장소로 바뀐 것과, 이를 모든 방문한 장소, 모든 user에 
대해 합한 것에 불과'''

## 먼저, 식1과 식2를 구현하는 함수가 필요.
''' 식1: log_data, parameter, i(or x_um)
'''

# data
''' data: [ user 0:   [ [i,r1,r2], [i,r1,r2],...,[i,r1,r2] ],
            user 1:   [ [i,r1,r2], [i,r1,r2],...,[i,r1,r2] ],
            ...
            user N-1: [ [i,r1,r2], [i,r1,r2],...,[i,r1,r2] ] ]
'''

import numpy as np
import math
import pdb

def p(data, data_loca, theta, phi, beta):
    # theta = N*Z (5*3)
    # phi = Z*I (3*5)
    
    # data = 
    # data_loca = 
    
    data = data.tolist() # numpy로 안짜서 일단 이렇게... ㅠ.ㅠ
    
    N = len(theta) # number of users
    Z = len(phi) # number of topics
    I = len(data_loca)
    
    p = [ [] for i in range(N) ]
    
    for u in range(N): # p[u]에 [len:z]를 M_u만큼 append
        
        i = 0
        for r_i_la, r_i_lo in data_loca: # data에 있는 모든 장소에 대해 계산필요
#         for x, r_la, r_lo in data[u]:
            p[u].append([]) # 여기에 담은 []에 topic 수만큼 식(2)를 계산
            
            for z in range(Z):
                R_u = [ data[u][idx][1:] for idx in range(len(data[u])) ]
#                 print(z, R_u)
                R_u = np.array(R_u) - np.array([r_i_la, r_i_lo])
#                 print(z, R_u)
                
                beta_term = np.sum(np.multiply(R_u, R_u))
#                 print(beta_term)
                beta_term = math.exp(-1/2*beta*beta_term)
#                 print(beta_term)
                p_x_z = math.exp(phi[z][i])*beta_term
                p[u][i].append(p_x_z)
                 
            i += 1
    
    p = np.array(p)
    p_sum = np.sum(p,axis=0)
    
    for idx, p_x in enumerate(p):
      # pdb.set_trace()
        p[idx] = np.divide(p_x,p_sum) #여기를 지나면 확률값이 같아짐
        
    return p 

###########################################################
[theta, phi] = initialize(5, 3, 5)

data = np.array([[[11,2,3],[12,3,3]],
                 [[22,3,3],[25,7,8],[21,2,3]],
                 [[31,2,3]],
                 [[43,5,3]],
                 [[53,5,3],[54,5,5]]])
data_loca = np.array([[2,3],[3,3],[5,3],[5,5],[7,8]])

beta = 2
p(data, data_loca, theta, phi, beta)

