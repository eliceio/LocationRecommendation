from topicmodel_class import TopicModel
import pandas as pd
import numpy as np # toy_data

import pdb

def load_data():
	'''
	Read data file
	File format: .csv, separated by tab
	'''

	# toy_data
	df = pd.DataFrame(np.array([ [int(1), int(1), 0.2, 0.8], 
								 [int(2), int(2), 0.8, 0.2], 
								 [int(3), int(1), 0.2, 0.8], 
								 [int(3), int(2), 0.8, 0.2] ]), 
					columns = ['Member ID','Restaurant ID','Restaurant Latitude','Restaurant Longitude'])

	# df = pd.read_csv('Daejeon_dataset_t.csv', delimiter='\t', index_col=False)
	return df


df = load_data()
print("Complete load data")

# pdb.set_trace() #1 

beta = 1 #float(input("Enter the beta value:")) #
Z = 2 #int(input("Enter the number of topic:")) #

N = len(df['Member ID'].unique())
I = len(df['Restaurant ID'].unique())

sys1 = TopicModel(df, beta, Z, N, I) #####################

# pdb.set_trace()

# training
beta, psi = sys1.trainParams(50) # 위 예제는 iteration 30에 학습 끝남.

## 최종 코드:
# [[0, 1],
#  [1, 0],
#  [0.5, 0.5]]
# [[-6, 6], [5, -5]]

## 바꾼 코드:
# [[0.012, 0.987],
#  [0.989, 0.010],
#  [0.174, 0.825]]
# [[-13, 13],[5, -5]]

## 원래 코드 :
# [[3.27e-06, 9.99e-01],
#  [8.75e-02, 9.12e-01],
#  [1.10e-04, 9.99e-01]]
# [[-8, 8],[5,-5]]

print(psi)
pdb.set_trace() #-1

# input test data 
current_location = input("Enter the current space:")
current_coordinate = sys1.get_location(current_location)

# test
recommend_prob = sys1.test(current_coordinate, psi, beta)

# print result
recommendation = sys1.find_recommendation(recommend_prob)
print(recommendation)


### change the file name loaded 

### Daejeon_dataset_t.csv
# Comment, Member ID, Member Nickname, Rating, Restaurant Address, Restuarant ID 
# Restaurant Latitude, Restaurant Longitude, Restaurant Nme, Restaurant code, 
# Restaurant subcode, Time  

### beta, Z optimization 