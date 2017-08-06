import urllib
import json
import sys
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

daejeon_list = ["동구 중앙동", "동구 신인동", "동구 효동", "동구 판암1동", "동구 판암2동", "동구 용운동", "동구 대동", "동구 자양동", "동구 가양1동", "동구 가양2동", "동구 용전동", "동구 성남동", "동구 홍도동", "동구 삼성동", "동구 대청동", "동구 산내동",
				"중구 운행선화동", "중구 목동", "중구 중촌동", "중구 대흥동", "중구 문창동", "중구 석교동", "중구 대사동", "중구 부사동","중구 용두동", "중구 오류동", "중구 태평1동", "중구 태평2동", "중구 유천1동", "중구 유천2동", "중구 문화1동", "중구 문화2동",
				"서구 복수동", "서구 도마1동", "서구 도마2동", "서구 정림동", "서구 변동", "서구 용문동", "서구 탄방동", "서구 둔산1동", "서구 둔산2동", "서구 둔산3동", "서구 괴정동", "서구 가장동", "서구 내동", "서구 갈마1동", "서구 갈마2동", "서구 월평1동", "서구 월평2동", "서구 월평3동", "서구 만년동", "서구 가수원동", "서구 관저1동", "서구 관저2동", "서구 기성동",
				"유성구 진잠동", "유성구 원신흥동", "유성구 온천1동", "유성구 온천2동", "유성구 노은1동", "유성구 노은2동", "유성구 노은3동", "유성구 신성동", "유성구 전민동", "유성구 구즉동", "유성구 관평동",
				"대덕구 오정동", "대덕구 대화동", "대덕구 회덕동", "대덕구 비래동", "대덕구 송촌동", "대덕구 중리동", "대덕구 법1동", "대덕구 법2동", "대덕구 신탄진동", "대덕구 석봉동", "대덕구 덕암동", "대덕구 목상동"]



def get_store(location):
    host_url = "https://www.mangoplate.com"
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
    comment_uuid = []
    restaurant_uuid = []
    restaurant_name = []
    restaurant_cusine = []
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
                comment_uuid.append(r["comment"]["comment_uuid"])
                restaurant_uuid.append(r["restaurant"]["restaurant_uuid"])
                restaurant_name.append(r["restaurant"]["name"])
                restaurant_cusine.append(r["restaurant"]["cusine_code"])
                restaurant_address.append(r["restaurant"]["address"])
                restaurant_lat.append(r["restaurant"]["latitude"])
                restaurant_long.append(r["restaurant"]["longitude"])
                
    return pd.DataFrame({'Restaurant ID': restaurant_uuid,
                         'Restaurant Name': restaurant_name,
                         'Restaurant Cusine': restaurant_cusine,
                         'Restaurant Address': restaurant_address,
                         'Restaurant Latitude': restaurant_lat,
                         'Restaurant Longitude': restaurant_long,
                        'Member ID': member_uuid, 
                        'Member Nickname': member_nick_name,
                        'Rating': review_rating,
                        'Time': reg_time,
                        'Comment': comment,
                        'Comment ID': comment_uuid})

if __name__ == "__main__":
	# search_place = input("Search Place : ")
	total_dataframe = pd.DataFrame()
	for search_place in daejeon_list:
		print("Start to get restaurant in " + search_place)
		search_place = "대전시 " + search_place
		location = urllib.parse.quote(search_place)	
		restaurants = get_store(location)
		data = get_review_set(restaurants)
		filename = "../data/crawl/" + search_place + ".csv"
		data.to_csv(filename)
		total_dataframe = pd.concat([total_dataframe, data], ignore_index=True)
	total_dataframe.drop_duplicates(cols="Comment ID")
	total_dataframe.to_csv("../data/crawl/Daejeon.csv")
	print("#######")
	print("#######")
	print("Finish")
	print("#######")
	print("#######")

