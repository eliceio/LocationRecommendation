from topicmodel_class import TopicModel
import pandas as pd

def load_data():
	'''
	Read data file
	File format: .csv, separated by tab
	'''
	df = pd.read_csv('Daejeon_dataset.csv', delimiter='\t', index_col=False)
	return df


df = load_data()
beta = float(input("Enter the beta value:"))
Z = int(input("Enter the number of topic:"))
N = len(df['Member ID'].unique())
I = len(df['Restaurant ID'].unique())

sys1 = TopicModel(df, beta, Z, N, I)

# training
beta, psi = sys1.trainParams(50)

# input test data 
current_location = input("Enter the current space:")
current_coordinate = sys1.get_location(current_location)

# test
recommend_prob = sys1.test(current_coordinate, psi, beta)

# print result
recommendation = sys1.find_recommendation(recommend_prob)
print(recommendation)