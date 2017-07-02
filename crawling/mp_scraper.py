import requests
from bs4 import BeautifulSoup, SoupStrainer
import urllib
import json
from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import csv


'''
주의사항: .csv에 append 가 되고있어서 다시 돌릴땐 이미 있는 csv파일 지우거나
        파일 이름을 바꿔서 돌리세요

        유성구만 검색되고 있습니닷
'''

def main():

    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # webpage address
    place_addr = []
    
    # csv format
    restaurant_name = []
    restaurant_address = []
    user_id = []
    user_rev =[]
    
    page = 1
    cnt = 1
    
    
    while page<11:
        # 유성구
        url = 'https://www.mangoplate.com/search/%EB%8C%80%EC%A0%84%20%EC%9C%A0%EC%84%B1%EA%B5%AC?keyword=%EB%8C%80%EC%A0%84%20%EC%9C%A0%EC%84%B1%EA%B5%AC&page=' + str(page)
    
        r = requests.get(url, headers=headers)

        soup = BeautifulSoup(r.text, 'lxml')

        # html print
        #print(soup.prettify())

        for link in soup.select('[class~=info] > a'):
            href = "www.mangoplate.com" + str(link.get('href'))
            title = link.find('h2')
        
            pass_addr = 'www.mangoplate.comNone'
        
            if href != pass_addr:
                place_addr.append(href)

        page += 1

    for i in range(len(place_addr)):
        # list clear
        restaurant_name.clear()
        restaurant_address.clear()
        user_id.clear()
        user_rev.clear()
        
        
        # test
        t_url = 'http://' + place_addr[i]
    
        # selenium 
        #driver = webdriver.Firefox()
        #driver.implicity_wait(5)
        #driver.get(t_url)


        req = urllib.request.urlopen(t_url)
        t_soup = BeautifulSoup(req, 'lxml')

        # restaurant_name
        r_name = t_soup.find("h1", class_="restaurant_name")

        # restaurant_address
        r_addr = t_soup.find("span", class_= "orange-underline")


        # extract user_info content
        user_data_list = t_soup.find_all('script', class_="review_menu_pictures_json" ,type="application/json")

        r_line = []
        
        for line in user_data_list:
            line = json.loads(line.text)
            if line not in r_line:
                r_line.append(line)

        for data in r_line:
            #restaurant_name
            name = r_name.get_text(strip = True)
            restaurant_name.append(name)

            #restaurant_address
            address = r_addr.get_text(strip = True)
            restaurant_address.append(address)
            
            # member_uuid
            uuid = data[0]['user']['member_uuid']
            user_id.append(uuid)
    
            # review
            rev = data[0]['review']['comment']
            rev = rev.replace('\n', '')
            user_rev.append(rev)
            
            # 돌아가는지 모르겠을때 주석을 푸세요 숫자가 찍힙니다.
            #print(cnt)
            cnt+=1


        dataset = pd.DataFrame({
            'restaurant_name': restaurant_name,
            'restaurant_addr': restaurant_address,
            'user_id': user_id,
            'user_review': user_rev
            })
        
        dataset.to_csv('유성구.csv', sep='\t', mode = 'a', index=False)
            
main()
