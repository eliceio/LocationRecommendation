'''
@author: Sumin Lim

Class version
'''

import numpy as np
import scipy as sp
from scipy.optimize import minimize


class SentimentRecommend():
    def __init__(self, num_user, num_loca, num_latent):
        self.N = num_user
        self.I = num_loca
        self.Z = num_latent


        # Set parameters
        self.U = np.zeros([self.N, self.Z])
        self.V = np.zeros([self.Z, self.I])


    def __initialize(self):
        self.U = np.random.random([self.N, self.Z])
        self.V = np.random.random([self.Z, self.I])

        return


    def __setTrainModuel(self):
        N = self.N
        I = self.I
        Z = self.Z


        
    def trainParams(self, maxiter, threshold, display=True):
        # Initialize
        self.__initialize()
        niter = 0

        self.__printLikelihood(niter, display)

        for niter in range(1, maxiter):

            condition = np.sqrt(np.sum(np.square(self.U - U)) + np.sum(np.square(self.V - V))) < threshold

            if (condition is True):
                break

            # Update parameters
            self.U, self.V = U, V
            self.__printLikelihood(niter, display)


        return
