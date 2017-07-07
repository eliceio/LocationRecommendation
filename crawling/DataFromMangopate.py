import urllib
import json
import sys
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

def get_store(location):
    host_url = "https://www.mangoplate.com"
    # location = "%EB%8C%80%EC%A0%84%20%EC%9C%A0%EC%84%B1%EA%B5%AC%20%EC%96%B4%EC%9D%80%EB%8F%99"
    restaurant = []
    pageNum = 0
    while True:
        pageNum += 1
        url = host_url + "/search/" +  location + "?" + "keyword=" + location + "&" + "page=" + str(pageNum)
        r = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(r, "html.parser")        
        stores = json.loads(soup.find('script', type="application/json", id="search_first_result_json").text.strip().replace("&quot;",'"'))
        
        if (len(stores) == 0):
            break
        
        for s in stores:
            restaurant.append(s["restaurant"]["restaurant_uuid"])

    print("The number of stores:", len(restaurant))
    return restaurant

def get_review_set(restaurants):
    member_uuid = []
    member_nick_name = []
    review_rating = []
    reg_time = []
    comment = []
    restaurant_uuid = []
    restaurant_name = []
    restaurant_address = []
    restaurant_lat = []
    restaurant_long = []
    
    for restaurant in restaurants:
    
        review_json_address = "https://stage.mangoplate.com/api/v5/restaurants/" + str(restaurant) + "/reviews.json?request_count=9999&start_index=0"

        # Test Address
        # review_json_address = 'https://stage.mangoplate.com/api/v5/restaurants/184801/reviews.json?access_token=&language=kor&request_count=99&start_index=0'
        
        review_json_data = json.loads(requests.get(review_json_address).text)
        for r in review_json_data:
                member_uuid.append(r["user"]["member_uuid"])
                member_nick_name.append(r["user"]["nick_name"])
                review_rating.append(r["action_value"])
                reg_time.append(r["time"])
                comment.append(r["comment"]["comment"])
                restaurant_uuid.append(r["restaurant"]["restaurant_uuid"])
                restaurant_name.append(r["restaurant"]["name"])
                restaurant_address.append(r["restaurant"]["address"])
                restaurant_lat.append(r["restaurant"]["latitude"])
                restaurant_long.append(r["restaurant"]["longitude"])
                
    return pd.DataFrame({'Restaurant ID': restaurant_uuid,
                         'Restaurant Name': restaurant_name,
                         'Restaurant Address': restaurant_address,
                         'Restaurant Latitude': restaurant_lat,
                         'Restaurant Longitude': restaurant_long,
                        'Member ID': member_uuid, 
                        'Member Nickname': member_nick_name,
                        'Rating': review_rating,
                        'Time': reg_time,
                        'Comment': comment})

if __name__ == "__main__":
	search_place = input("Search Place : ")
	location = urllib.parse.quote(search_place)
	restaurants = get_store(location)
	data = get_review_set(restaurants)
	filename = search_place + ".pkl"
	data.to_pickle(filename)

