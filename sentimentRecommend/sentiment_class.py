'''
@author: Sumin Limm, Hyunji Lee

Class version
'''

import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, logistic
from scipy.special import expit
import pdb

def get_log_posterior(U, V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z):
    
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



class SentimentRecommend():
    '''
    self 로 처리할 것들:
       df, U, V, N, Z, I
        pref_final, sim_u, sim_v,
        lambda_v, lambda_u
        alpha, beta
        train, test

        self.U, self.V, self.train, self.sim_u, self.sim_v, self.lambda_u, self.lambda_v, self.alpha, self.beta, self.N, self.I, self.Z
    ''' 
    def __init__(self, df, num_latent):

        self.df = df
        self.Z = num_latent

        # # Set parameters
        # self.U = np.zeros([self.N, self.Z])
        # self.V = np.zeros([self.Z, self.I])

    def __initialize(self):
        # input: self.df
        # output: self.pref_checkin, self.pref_sentiment
        self.__get_pref_mats()

        # input: self.pref_checkin, self.pref_sentiment
        # output: self.pref_final
        self.__compute_pref_final()

        # input: self.pref_final
        # ouput: train, test
        self.__split_train_test()

        self.N, self.I = self.pref_final.shape
        
        print("initialize U, V")
        self.U = np.random.rand(self.N, self.Z)
        self.V = np.random.rand(self.Z, self.I)

        # input: self.pref_final
        # output: self.sim_u, self.sim_v
        print("get similarity of U, V")
        self.__get_sim_u()
        self.__get_sim_v()

        # all self.
        # input: pref_final, sim_u, sim_v, U, V
        # output: lambda_u, lambda_v, alpha, beta
        self.__get_coefficient()


    def __setTrainModuel(self):
        N = self.N
        I = self.I
        Z = self.Z

    def __get_pref_mats(self):

        print("making preference matrices")

        mem_id = sorted(self.df['Member ID'].unique()); loc_id = sorted(self.df['Restaurant ID'].unique())
        pref_checkin = pd.DataFrame(0, index=mem_id, columns=loc_id)
        pref_sentiment = pd.DataFrame(0, index=mem_id, columns=loc_id)
        
        for index, row in self.df.iterrows():
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

        self.pref_checkin = np.array(pref_checkin) 
        self.pref_sentiment = np.array(pref_sentiment)
        

    def __compute_pref_final(self):
        print("making final preference matrix")
        self.pref_final = self.pref_checkin - np.sign(self.pref_checkin - self.pref_sentiment) * np.heaviside(np.abs(self.pref_checkin - self.pref_sentiment)-2, 0.5)
    
    def __get_sim_u(self):

        print("__get_sim_u")
        N, _ = self.pref_final.shape
        sim_u = []
        #pdb.set_trace()

        cnt = 1
        for n in range(N):
            #loop log
            if n%(int(N/10)) == 0 and n!=0:
                print("get_sim_u_complete: {:d} %".format(cnt*10))
                cnt += 1

            temp = []
            for i in range(N):
                temp.append(pearsonr(self.pref_final[n], self.pref_final[i])[0])
            sim_u.append(temp)

        self.sim_u = np.array(sim_u)


    def __get_sim_v(self):

        print("__get_sim_v")
        location = []
        for index, row in self.df.iterrows():
            tmp = [row['Restaurant ID'], row['Restaurant code']]
            if tmp not in location:
                location.append(tmp)

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
        self.sim_v = sim_v

    def __get_coefficient(self):
        print("get coefficient")
        var_R = np.var(self.pref_final)
        self.lambda_u = var_R / np.var(self.U)
        self.lambda_v = var_R / np.var(self.V)
        self.alpha = var_R / np.var(self.sim_u)
        self.beta = var_R / np.var(self.sim_v)
    
    def __split_train_test(self):
        print("making training set and test set")
        test = np.zeros(self.pref_final.shape)
        train = self.pref_final.copy()
        N = self.pref_final.shape[0]
        for user in range(N):
            visited = self.pref_final[user, :].nonzero()[0]
            if len(visited) == 1:
                pass
            elif len(visited) > 3:
                test_pref = np.random.choice(visited, size=3, replace=False)
                train[user, test_pref] = 0.
                test[user, test_pref] = self.pref_final[user, test_pref]

        # Check train and test set is independent
        assert (np.all((train * test) == 0))
        
        self.train = train 
        self.test = test



    def trainParams(self, maxiter, threshold, display=True):
        # Initialize
        self.__initialize()
        
        niter = 0

        #self.__print_posterior(niter, display)

        cnt = 0
        for niter in range(1, maxiter):
            cnt += 1
            print("==================== In the While Loop =======================")
            print(" %d th iteration" % cnt)

            u_res = minimize(get_log_posterior,x0 = self.U, args = (self.V, self.train, self.sim_u, self.sim_v, self.lambda_u, self.lambda_v, self.alpha, self.beta, self.N, self.I, self.Z),jac = get_grad_u)

            pdb.set_trace()

            v_res = minimize(get_log_posterior,x0 = self.V, args = (self.U, self.train, self.sim_u, self.sim_v, self.lambda_u, self.lambda_v, self.alpha, self.beta, self.N, self.I, self.Z),jac = get_grad_v)

            estimated_U = u_res.x.reshape(self.N, self.Z)
            estimated_V = v_res.x.reshape(self.Z, self.I)


            condition = np.sqrt(np.sum(np.square(self.U - estimated_U)) + np.sum(np.square(self.V - estimated_V))) < threshold

            if (condition is True):
                break

            # Update parameters
            self.U, self.V = estimated_U, estimated_V
            
            # 이부분 돌아가는지 확인해 
            #self.__print_posterior(niter, display)

            # Update parameters: lambda_u, lambda_v, alpha, beta
            self.__get_coefficient()

        ####### Utils to print likelihood for each iteration #######
    
        MAE, RMSE = compute_metrics(self.U, self.V, self.test)
        print("\n\n=========================================================")
        print("testing")
        print("MAE:", MAE)
        print("RMSE:", RMSE)
    

    #def get_log_posterior(U, V, pref_final, sim_u, sim_v, lambda_u, lambda_v, alpha, beta, N, I, Z)

    def __computeLikelihood(self):
        U = sp.resize(self.U, (self.N, self.Z))
        V = sp.resize(self.V, (self.Z, self.I))
        first_term = np.sum(self.pref_final - expit(U @ V))
        second_term = self.lambda_u * np.sum(U @ U.T) + self.lambda_v * np.sum(V @ V.T)
        third_term = self.alpha * np.sum((U - (self.sim_u @ U)) @ (U - (self.sim_u @ U)).T)
        fourth_term = self.beta * np.sum((V.T - (self.sim_v @ V.T)) @ (V.T - (self.sim_v @ V.T)).T)
        log_posterior = 0.5 * (first_term + second_term + third_term + fourth_term)
        
        return log_posterior

    # posterior == likelihood
    def __print_posterior(self, niter, display):
        if display is True:
            L = self.__computeLikelihood()
            if niter is 0:
                print('Initial Log Posterior : %.3f' % L)
            else:
                print('Iter : %d, L value : %.3f' % (niter, L))
