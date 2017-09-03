#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:54:40 2017

@author: vinnam
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import pearsonr

import pdb

def maximizingPhi(Phi, sess, feed_dict, Phi_tensor, outputs, Z, I):
        feed_dict[Phi_tensor] = Phi.reshape([Z, I])
        
        Q, grad = sess.run(outputs, feed_dict)
        
        return Q, grad.reshape([-1])
    
class GeoTopic():
    def __init__(self, num_user, num_loca, num_topic, beta, user_log, loc_info):
        # num_user = N
        # num_loca = I
        # num_topic = Z
        # beta is a parameter beta
        self.N = num_user
        self.I = num_loca
        self.Z = num_topic
        self.beta = beta
        
        # user_log is a (M_1 + ... + M_N) x 2 of 2D numpy array 
        # Each row is [user, location]
        # For example, [[1, 4], [1, 5], ..., [N, 100]]
        _user_log = user_log - 1
        
        self.user_log = np.concatenate([np.append(z * np.ones([_user_log.shape[0], 1]), _user_log, axis = 1) for z in range(self.Z)], axis = 0)
        
        # loc_info is a I x 2 of 2D numpy array
        # Each row is [longitude, latitude] of a location
        # For example, [..., [123.3, 54.5], ...], [123.3, 54.5] has 5th place means geographic information of the location 5
        
        self.loc_info = loc_info
        
        # N x I
        # self.user_log_mat : Sparse incidence matrix for (user x location)
        # self.beta_dists_sum : Compute \sum_{r \in R_u} exp(-0.5 * beta ||r_i - r||^2)
        
        ################## Modify these codes to use Tensorflow for computing beta_dists ##################
        entries = np.ones(shape = _user_log.shape[0])
        rowcol = (_user_log[:, 0], _user_log[:, 1])
        
        self.user_log_mat = csr_matrix((entries, rowcol), shape = (self.N, self.I))
        
        beta_dists = np.exp(-0.5 * beta * np.square(squareform(pdist(self.loc_info))))
        
        self.beta_dists = self.user_log_mat.dot(beta_dists)
        ###################################################################################################
        
        # Set parameters
        self.Theta = np.zeros([self.N, self.Z])
        self.Phi = np.zeros([self.Z, self.I])
        
        self.sess_train, self.inputs_train, self.outputs_train = self.__setTrainModule()
        
    ####### Set Tensorflow computation module to train model #######
    
    def __setTrainModule(self):
        N = self.N
        I = self.I
        Z = self.Z
        
        graph = tf.Graph()
        
        with graph.as_default():
            # Parameteres
            Theta = tf.placeholder(tf.float64, shape = [N, Z])
            Phi = tf.placeholder(tf.float64, shape = [Z, I])
            
            # Summation along users with exponentiated distances between locations
            beta_dists = tf.placeholder(tf.float64, shape = [N, I])
            
            # Visited logs
            z_user_log = tf.placeholder(tf.int64)
            
            # Z_posterior_values is the values of the equation (5) which requires for computing Q, equation (4)
            Z_posterior_values = tf.placeholder(tf.float64)
            
            # Sparse tensor of Z_posterior_values
            Z_posterior = tf.SparseTensor(indices = z_user_log, values = Z_posterior_values, dense_shape = [Z, N, I])
            
            # Sparse tensor of an indicating matrix
            Indices = tf.SparseTensor(indices = z_user_log, values = tf.ones([tf.shape(z_user_log)[0]], dtype = tf.float64), dense_shape = [Z, N, I])
            
            
            # Z x 1 x I
            P_numer_front = tf.expand_dims(tf.exp(Phi), axis = 1)
            # 1 x N x I
            P_numer_back = tf.expand_dims(beta_dists, axis = 0)
            # Z x N x I
            P_numer = P_numer_front * P_numer_back
            # Z x N x 1
            P_denom = tf.expand_dims(tf.reduce_sum(P_numer, axis = 2), axis = 2)
            # Z x N x I
            # P is the equation (2)
            P = tf.div(P_numer, P_denom, name = 'P_i')
            # Z x N x 1
            Theta_exp = tf.expand_dims(tf.transpose(Theta), axis = 2)
            
            
            # Q is the eqaution (4)
            Q = Z_posterior * tf.log(Theta_exp * P)
            # For minimization solver
            Q = tf.negative(tf.sparse_reduce_sum(Q), name = 'Q')
            
            # Phi_grad is the equation (7)
            Phi_grad = tf.gradients(Q, Phi)
            # This is because tensorflow makes Phi_grad as a list of a tensor : [tf.gradients]
            Phi_grad = Phi_grad[0]
            
            # P_z is the equation (5)
            P_z = tf.expand_dims(tf.transpose(Theta), axis = 2) * P
            
            P_z = Indices * P_z
        
            P_z = P_z / tf.sparse_reduce_sum(P_z, axis = 0)
            
            # Z x N
            Theta_hat = tf.sparse_reduce_sum(Z_posterior, axis = 2)
            
            Theta_hat = Theta_hat / tf.reduce_sum(Theta_hat, axis = 0)
            
            # N x Z
            # Theta_hat is the equation (6)
            Theta_hat = tf.transpose(Theta_hat, name = 'Theta_hat')
            
            # Likelihood
            L = tf.expand_dims(tf.transpose(Theta), axis = 2) * P
            L = tf.log(tf.reduce_sum(L, axis = 0))
            L = tf.sparse_reduce_sum(Indices, axis = 0) / Z * L
            L = tf.reduce_sum(L)

        sess = tf.Session(graph = graph)
        
        inputs ={'Theta' : Theta,
                'Phi' : Phi,
                'beta_dists' : beta_dists,
                'z_user_log' : z_user_log,
                'Z_posterior_values' : Z_posterior_values}
                
        outputs = {'Q' : Q,
                   'Phi_grad' : Phi_grad,
                   'P' : P,
                   'P_z' : P_z,
                   'Theta_hat' : Theta_hat,
                   'L' : L}
                
        return sess, inputs, outputs
    
    ####### Expectation - Maximization algorithm ########
    
    def __initialization(self, std = .1):
        # Randomly initialize
        self.Theta = np.random.random([self.N, self.Z])
        
        # Normalize by constraints
        self.Theta = self.Theta / np.sum(self.Theta, axis = 1).reshape([-1, 1])
        
        # Randomly initialize with a Gaussian dist
        self.Phi = np.random.normal(loc = 0.0, scale = std, size = [self.Z, self.I])
        
        return
    
    def __Estep(self):
        # Obtain the posterior distribution of Z|u, m
        sess = self.sess_train
        tensor = self.outputs_train['P_z']
        feed_dict = {self.inputs_train['Theta'] : self.Theta,
                      self.inputs_train['Phi'] : self.Phi,
                      self.inputs_train['beta_dists'] : self.beta_dists,
                      self.inputs_train['z_user_log'] : self.user_log}
        
        return sess.run(tensor, feed_dict).values
    
    def __Mstep(self, Z_posterior):
        # Optimize Theta
        sess = self.sess_train
        
        Theta_tensor = self.outputs_train['Theta_hat']
        
        feed_dict = {self.inputs_train['z_user_log'] : self.user_log,
                      self.inputs_train['Z_posterior_values'] : Z_posterior}
        
        Theta_hat = sess.run(Theta_tensor, feed_dict)
        
        # Optimize Phi
        x0 = self.Phi.reshape([-1])
        
        feed_dict = {self.inputs_train['Theta'] : Theta_hat,
                      self.inputs_train['Phi'] : self.Phi,
                      self.inputs_train['beta_dists'] : self.beta_dists,
                      self.inputs_train['z_user_log'] : self.user_log,
                      self.inputs_train['Z_posterior_values'] : Z_posterior}
        
        outputs = [self.outputs_train['Q'], self.outputs_train['Phi_grad']]
        
        Phi_tensor = self.inputs_train['Phi']
        
        Z = self.Z
        I = self.I
        
        res = minimize(maximizingPhi, x0, jac = True,
                        args = (sess, feed_dict, Phi_tensor, outputs, Z, I,))
        
        Phi_hat = res.x.reshape(Z, I)
        
        # Obtain Q function value
        Q = res.fun
        
        return Theta_hat, Phi_hat, Q
        
    def trainParams(self, maxiter, threshold, display = True):
        # Initialization
        self.__initialization()
        niter = 0
        
        self.__printLikelihood(niter, display)
        
        # E and M steps until converge
        for niter in range(1, maxiter):
            # Obtain the posterior distributions of Z|u, m
            Z_posterior = self.__Estep()
            
            # Obtain the next parameters
            Theta, Phi, Q = self.__Mstep(Z_posterior)
            
            # If Euclidean norm of the difference between gradient vectors
            # is less than threshold, terminate
            condition = np.sqrt(np.sum(np.square(self.Theta - Theta)) + \
                                np.sum(np.square(self.Phi - Phi))) < threshold
            
            if(condition is True):
                break
            
            # Update parameters
            self.Theta, self.Phi = Theta, Phi
            self.__printLikelihood(niter, display)
            
        return
    
    ####### Utils to print likelihood for each iteration #######
    
    def __computeLikelihood(self):
        sess = self.sess_train
        tensor = self.outputs_train['L']
        feed_dict = {self.inputs_train['Theta'] : self.Theta,
                      self.inputs_train['Phi'] : self.Phi,
                      self.inputs_train['beta_dists'] : self.beta_dists,
                      self.inputs_train['z_user_log'] : self.user_log}
        
        return sess.run(tensor, feed_dict)
        
    def __printLikelihood(self, niter, display):
        if display is True:
            L = self.__computeLikelihood()
            if niter is 0:
                print('Initial Likelihood : %.3f' % L)
            else:
                print('Iter : %d, L value : %.3f' % (niter, L))
        return
    
    #############################################################
    
    def recommendNext(self, user_id, current_loc):
        # indices should start from zero
        indices = user_id - 1
        # U x Z
        Theta = self.Theta[indices]
        # Z x I
        Phi = self.Phi
        beta = self.beta
        
        # U x I
        dists = np.exp(-0.5 * beta * np.square(cdist(current_loc, self.loc_info)))
        
        # Equation (8)
        # Z x U x I
        P_izu = np.expand_dims(np.exp(Phi), axis = 1) * np.expand_dims(dists, axis = 0)
        P_izu = P_izu / np.expand_dims(np.sum(P_izu, axis = 2), axis = 2)
        
        # Equation (9)
        # U x I
        P_iu = np.expand_dims(Theta.T, axis = 2) * P_izu
        P_iu = np.sum(P_iu, axis = 0)
        
        return P_iu

    def recommendNext_sim(self, user_idx):
        user_vector = self.Theta[user_idx]
        loc_vector = self.Phi.transpose()

        # Euclidean distance user_vector, loc_vector
        z = loc_vector - user_vector 
        z = np.power(z,2)
        z = np.sqrt(np.sum(np.power(z,2), axis=1)).tolist() # 거리 list

        # print('sim')
        # pdb.set_trace()
        recommend = np.array(sorted(range(len(z)), key=lambda k:z[k]))
        recommend += 1

        return (recommend, recommend[::-1]) #sorted(range(len(z)), key=lambda k:z[k])
        # 작을 수록 추천!, 클수록 반대로 추천! 

    def recommendNext_cos(self, user_idx):
        user_vector = self.Theta[user_idx] # 1*Z
        loc_vector = self.Phi.transpose() # I*Z

        z = np.inner(user_vector, loc_vector)
        z = z/np.linalg.norm(user_vector) # np.linalg.norm(user_vector) 는 상수
        z = z/np.linalg.norm(loc_vector, axis=1) # np.linalg.norm(loc_vector) 는 1*I

        # print('cos')
        # pdb.set_trace()
        recommend = np.array(sorted(range(len(z)), key=lambda k:z[k]))
        recommend += 1
        return (recommend[::-1], recommend) #sorted(range(len(z)), key=lambda k:z[k])
        # 클수록 추천! 

    def recommendNext_pear(self, user_idx):
        user_vector = self.Theta[user_idx]
        loc_vector = self.Phi.transpose()

        z = []
        for loc in loc_vector:
            corr, p_value = pearsonr(user_vector, loc)
            z.append(corr)

        recommend = np.array(sorted(range(len(z)), key=lambda k:z[k]))
        recommend += 1
        return (recommend[::-1], recommend)
        # 클수록 추천! 

    def measure(self, recommend, test_data):
        pre_at_N = []
        for i in range(self.I):
            # pdb.set_trace()
            recommend_at_N = recommend[:,:(i+1)] # N * (i+1)

            accuracy = 0
            for n in range(self.N):
                if test_data[n][1] in recommend_at_N[n]:
                    accuracy += 1
            accuracy = accuracy/self.N*100

            pre_at_N.append(accuracy)

        MRR = 0
        for n in range(self.N):
            t_recommend = recommend.tolist() # N*I
            # pdb.set_trace()
            mrr_index = t_recommend[n].index(test_data[n][1])+1
            MRR += 1/mrr_index
            MRR = MRR/self.N

        return pre_at_N, MRR

    # def pre_at_N(self, recommend, test_data):
    #     pre_at_N = []
    #     for i in range(self.I):
    #         # pdb.set_trace()
    #         recommend_at_N = recommend[:,:(i+1)] # N * (i+1)

    #         accuracy = 0
    #         for n in range(self.N):
    #             if test_data[n][1] in recommend_at_N[n]:
    #                 accuracy += 1
    #         accuracy = accuracy/self.N*100

    #         pre_at_N.append(accuracy)   
        
    #     return pre_at_N
    
    # def MRR(self, recommend, test_data): # recommend는 numpy
    #     MRR = 0
    #     for n in range(self.N):
    #         t_recommend = recommend.tolist() # N*I
    #         # pdb.set_trace()
    #         mrr_index = t_recommend[n].index(test_data[n][1])+1
    #         MRR += 1/mrr_index
    #         MRR = MRR/self.N

    #     return MRR
