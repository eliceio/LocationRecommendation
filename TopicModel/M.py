import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform # distance.euclidean(a,b)
import tensorflow as tf

def getDistance(beta, location):
    '''
    '''
    # location: 데이터셋에 있는 모든 장소
    # dist: 모든 장소에 대해 거리 계산한 매트릭스, I*I
    I = len(location)
    dist = squareform(pdist(np.exp(-0.5*beta*location)))
    
    return dist
            

# Equation 2
def p_i(beta, phi, dist, x_um):
    '''
    (2)번 계산
    '''
    # x_um : 유저가 방문한 장소의 index list. len(x_um): Mu
    # res: P(i|z, R_u, phi). Z*I 
    z = phi.shape[0]
    Mu = len(x_um)
    userDist = dist[:, x_um]
    userDistSum = userDist.sum(axis=1)
    
    phi = np.exp(phi)
    res = phi * userDistSum

    return res

def E(psi, beta, dist, X):
    # X: 모든 유저가 방문한 장소의 index list. X[0] => 0번째 유저가 방문한 장소의 inde list
    theta = psi[0]; phi = psi[1]
    for user in X:
        px_um = p_i(beta, phi, dist, user)
        theta[user]
#        theta[u,:] * px_um => should be (Z*I) shape
#        P(z|u, m; psi) = theta[u,:] * px_um / sum(px_um).axis=0

def C(phi, location, user_locs):
    # 모든 유저에 대해서, 모든 장소에 대해서 C 계산
    # E-step에서는 필요 없지만 M-step에서는 필요함

def fun(probs, theta, P):
    '''
    This function returns Q and its partial derivatives w.r.t phi_z
    input x should be an ndarray, and should contain P(z|u,m;psi), theta_uz, P(x_um|z,R_u, phi)
    '''
    # probs P(z|u,m; psi): Z*I ndarray가 N개 있음
    # P = P(x_um|z,R_u, phi): Z*I
    # theta => N*Z

    # Q => scalar
    Q = np.sum(probs * np.log(theta) * P)
    # grad_Q => I*1 vector 가 Z 개 있음
    grad_Q = def p_i(beta, phi, dist, x_um):
    return Q, grad_Q


def M(psi, probs, X, beta):
    ''' 
    Input explanation:
    1. psi: (theta, phi) or [theta, phi]
    2. probs P(z|u, m; psi) => len(probs) = N, probs[0].shape = Z * Mu
    3. r_i: geotag of location i represented by latitude and longitude coordinates in bi-axial space
    4. R_u: the set of geotags of locations of user u
    5. beta: User input

    Output: psi = (theta, phi)
    ''' 

    # Optimize \theta_uz 
    theta = psi[0]; phi = psi[1]; N = len(probs)
    thetaHat = []
    for prob in probs:
        Msum = prob.sum(axis=1)
        Zsum = Msum.sum()
        thetaHat.append(Msum / Zsum)

    thetaHat = np.array(thetaHat)

    # Optimize phi_z with a gradient-descent method
    # P = P(x_um|z,R_u, phi)
    # for user in X:
    #     px_um = p_i(beta, phi, dist, user)
    #     Q, grad_Q = fun(probs, thetaHat, )

    P_hat = tf.placeholder(tf.float64, shape=[Z, N, I])
    Theta = tf.placeholder(tf.float64, shape=[N, Z])
    Phi = tf.placeholder(tf.float64, shape=[Z, I])
    dist = tf.placeholder(tf.float64, shape=[I, I])

    idx = tf.SparseTensor(indices=indices, values=tf.ones(len(indicies), dtype=tf.float64), dense_shape=[N, I])

    front = tf.exp(Phi)
    back = tf.sparse_tensor_dense_matmul(idx, dist)

    P_numer = tf.expand_dims(front, axis=1) * back
    P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis=2), axis=2)
    P = P_numer / P_denom

    log_theta = tf.expand_dims(tf.transpose(tf.log(Theta)), axis=2)
    loglike = P_hat * log_theta * P
    Q = tf.reduce_sum(tf.sparse_tensor_dense_matmul(idx, tf.transpose(tf.reshape(loglike, [-1, I]))))

    Phi_grad = tf.gradients(Q, phi)
    minimize(objective, jac = gradient)
    
    return psi


# def E(psi, beta, user_locs, location):
#     '''
#     '''
#     # user_loc: 유저가 방문한 장소의 index list 의 list
#     # location: 데이터 셋 안에 있는 모든 장소 (경도, 위도)
#     # output: P(z|u,m; phi)
#     theta = psi[0]; phi = psi[1]
#     N = theta.shape[0]; Z = theta.shape[1]; I = phi.shape[1]

    
    
#     def p_i(beta, phi, dist, x_um):
P_hat = tf.placeholder(tf.float64, shape = [Z, N, I])
Theta = tf.placeholder(tf.float64, shape = [N, Z])
Phi = tf.placeholder(tf.float64, shape = [Z, I])
dist = tf.placeholder(tf.float64, shape = [I, I])

# Z = 5
# N = 10
# I = 20


# indices is tf tensor
# [[user, location index], [user, location index], .....]
# indices = []
# for i in range(N):
#     for j in range(np.random.randint(1, 5)):
#         indices.append([i, np.random.randint(0, I)])

# 만약 user 0 이 방문한 장소의 인덱스가 1,3,5,7이다 => indices = [[0,1], [0,3], [0,5], [0,7]]
Indices = tf.SparseTensor(indices = indices, values = tf.ones(len(indices), dtype = tf.float64), dense_shape = [N, I])


# P_hat * Indices : Z x N x I

# Z x I
front = tf.exp(Phi)

# N x I
back = tf.sparse_tensor_dense_matmul(Indices, dist)

P_numer = tf.expand_dims(front, axis =1) * back

P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis = 2), axis = 2)

P = P_numer / P_denom

log_Theta = tf.expand_dims(tf.transpose(tf.log(Theta)), axis = 2)

loglike = P_hat * log_Theta * P

Q = tf.reduce_sum(tf.sparse_tensor_dense_matmul(Indices, tf.transpose(tf.reshape(loglike, [-1, I]))))

Phi_grad = tf.gradients(Q, Phi)



#========================================================
def func(phi, theta, P_hat, dist):
    feed_dicc = {P_hat : P_hat,
                 Theta : np.random.random([N, Z]),
                 Phi : phi,
                 dist : np.random.random([I, I])}
    
   Q_ = sess.run(Q, feed_dict)

   Phi_grad_ = sess.run(Phi_grad, feed_dict)
    
   return Q_, Phi_grad_

fun = lambda(phi : func(phi, theta, P_hat, dist))
minimize(fun, jac = True)


def objective(phi):
    phi.reshape([Z, I])
    Q_ = sess.run(Q, feed_dict)
    return Q_

def gradient():
    Phi_grad_ = sess.run(Phi_grad, feed_dict)
    return Phi_grad_.reshape([-1])

## Tensorflow input phi Z x I
## Scipy optimizer input phi 1D array

minimize(objective, jac = gradient)
