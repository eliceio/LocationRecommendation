import json
import urllib.request
from urllib import parse
import pandas as pd
from multiprocessing import Pool,cpu_count
import time
import numpy as np
import pdb


def get_location(current_location):
    current_address = parse.quote(str(current_location))
    address = urllib.request.urlopen("http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&latlng=" + current_address).read().decode('utf-8')

    data = json.loads(address)
    
    try:
        location_type = data["results"][0]["types"]
    except IndexError:
        return None
        
    return location_type

def work(data):
    loc_type = []
    for i in range(len(data)):
        current_location = str(data[i][0]) + ',' +  str(data[i][1])

        #pdb.set_trace()

        loc = get_location(current_location)
        
        if loc == ['premise']:
            loc_type.append(i)

        print(i)

    return loc_type

def main():
    cores = 20 #cpu_count()여야 하는데...
    partitions = cores
    
    start_time = time.time()
    df = pd.read_csv('geo21',index_col=False, usecols = ["Latitude", "Longitude", 'UserId'])
    
    # user : 20번까지 있었음
    user_0 = df.loc[df['UserId'] == 0]
    
    coord = user_0.values.tolist()
    a = np.array_split(coord, partitions)
    
    #print(a[9])
    #p = Pool(cores)
    #while
    data = work(a[0])


    #data = parallelize(coord, work)

    my_df = pd.DataFrame(data) 
    my_df.to_csv('user_1_out.csv', index=False, header=False)
    
    #my_df.to_csv('out.csv','a', index=False, header=False)

    end_time = time.time()
    print("time:", end_time - start_time)

if __name__ == '__main__':
	main()