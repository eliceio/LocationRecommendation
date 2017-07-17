import numpy as np
import scipy.optimize import minimize

def gradient():
    '''
    This function returns Q and its partial derivatives w.r.t phi_z
    input x should be an ndarray, and should contain P(z|u,m;psi), theta_uz, P(x_um|z,R_u, phi)
    '''
    
    Q =
    return Q, partialQ

def M(psi, probs, r_i, R_u, beta):
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

    thetaHat = np.array(optTheta)

    # Optimize phi_z with a gradient-descent method
    
    return psi

