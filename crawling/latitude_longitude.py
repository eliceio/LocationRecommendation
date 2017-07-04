import pandas as pd
import json
import urllib.request
from urllib import parse
import json

# csv 파일은 원하는걸루 바꾸세요.
# dataset.csv 파일은 sep = ',' 으로 바꿔야합니다.
df = pd.read_csv('seogu.csv', sep='\t', usecols=['store_location'])
store_loc = df.values

for location in store_loc:
	# 한글파라미터 전송시
	location = parse.quote(str(location))

	# url
	addr = urllib.request.urlopen("http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address=" + location).read()
	
	data = json.loads(addr)
	latitude = data["results"][0]["geometry"]["location"]["lat"]
	longitude = data["results"][0]["geometry"]["location"]["lng"]

	print(latitude, longitude)


'''
location = '대전시 서구 복수동 475'
location = parse.quote(location)

addr = urllib.request.urlopen("http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address=" + location).read()
	
json = json.loads(addr)
latitude = json["results"][0]["geometry"]["location"]["lat"]
longitude = json["results"][0]["geometry"]["location"]["lng"]

print(latitude)
print(longitude)
'''
