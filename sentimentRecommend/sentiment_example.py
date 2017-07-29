from sentiment_class import SentimentRecommend
import pandas as pd

def load_data():
    '''
    load data. This file uses temp data (daejeon.csv)
    '''
    df = pd.read_csv('Daejeon_dataset.csv', delimiter='\t', index_col=False)
    return df


Z = int(input("Input the number of latent space:")) 
df = load_data()

sys1 = SentimentRecommend(df, Z)

sys1.trainParams(50, 1e-6)