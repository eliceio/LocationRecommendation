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