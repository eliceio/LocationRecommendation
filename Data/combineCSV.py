import os
import sys
import pandas as pd


def main():
    path = input("Input the data location path:")
    df = []
    shapeCheck = []

    for filename in os.listdir(path):
        if filename != "data_collect.py":
            temp = pd.read_csv(filename, delimiter='\t', index_col=False)
            print(temp.shape)
            shapeCheck.append(temp.shape[0])
            df.append(temp)

    print("The number of columns of each data file:", shapeCheck)
    print("The number of columns of concated data should be:", sum(shapeCheck))
    dataset = pd.concat(df)
    print("The number of columns of concated data:", dataset.shape[0])

    filename = input("Input your filename to export dataset excluding .csv:")
    filename = filename + '.csv'
    dataset.to_csv(filename, sep='\t', index=False, encoding='utf-8')

if __name__=="__main__":
    main()
