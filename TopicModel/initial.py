import pandas as pd
import numpy as np

# Algorithm 2
def initialize(nUser, nTopic, nLoc):

    theta = np.random.rand(nUser, nTopic)
    col_sum = theta.sum(axis=0)
    theta = theta / col_sum

	phi = np.array()
	phi = np.random.rand(nTopic, nLoc)


	return [theta, phi]


def Estimation():
	nUser = None
	nTopic = None
	nLoc = None

	# Initialize parameters psi = {theta, phi} (Algorithm 2)
	theta, phi = initialize(nUser, nTopic, nLoc)

	# Set the value of max_iter, beta arbitrarily
	max_iter = 1000

	for i in range(max_iter):
		# Proceed E-step
		# Proceed M-step
		theta, phi = algo4(theta, phi, algo3(theta, phi))
		
		if converge(theta, phi) == True:
			return

	return