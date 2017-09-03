#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:42:49 2017

@author: vinnam
"""
### Import class of the model
from GEOTOPIC import GeoTopic
import numpy as np

# Example toy data
# 3 users, 2 locations, 2 topics
num_user = 3
num_loca = 2
num_topic = 2
beta = 100
user_log = np.array([[1, 1],
                     [2, 2],
                     [3, 1],
                     [3, 2]])
loc_info = np.random.random(size = [2, 2])

### Create Model instance
sys1 = GeoTopic(num_user, num_loca, num_topic, beta, user_log, loc_info)

### Train Model
# Max iteration = 50
# Convergence threshold = 0.1
sys1.trainParams(50, 0.1)

# Print params
### In convergence, parameters would be
### Theta = [[0, 1], [1, 0], [0.5, 0.5]]
### Phi = [[postiive, negative], [negative, positive]]
print(sys1.beta_dists)
print(sys1.Theta)
print(sys1.Phi)

### Recommend next locations

# Inputs
user_id = np.array([3, 3])
current_loc = loc_info

### Obtain probabilities of the next locations
probs = sys1.recommendNext(user_id, current_loc)

# Print probabilities
print(probs)