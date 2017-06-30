import urllib.request
import json
import sys
import pandas as pd
from bs4 import BeautifulSoup


def get_store():
    '''
    input: host_url ("https://www.mangoplate.com"), location ("대전 유성구 어은동", this should be converted into unicode)
    output: list containing review site (['https://www.mangoplate.com/restaurants/JVom8cKaKsBo', ... ])
    '''
    host_url = "https://www.mangoplate.com"
    location = "%EB%8C%80%EC%A0%84%20%EC%9C%A0%EC%84%B1%EA%B5%AC%20%EC%96%B4%EC%9D%80%EB%8F%99"
    restaurant = []
    pageNum = 0
    while True:
        pageNum += 1
        url = host_url + "/search/" +  location + "?" + "keyword=" + location + "&" + "page=" + str(pageNum)
        r = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(r)
        is_empty = True
        for link in soup.find_all('a'):
            href = link.get('href')
            if (href != None) and (href.startswith('/restaurants/')):
                temp = host_url + href
                is_empty = False
                if temp not in restaurant:
                    restaurant.append(temp)
        if (is_empty):
            break

    return restaurant

def makingDataset(restaurant):
    '''
    input: filename (어은동.txt), restaurant (list of links)
    '''
    str_name = []; str_loc = []; str_rate = []; usr_id = []; usr_time = []; usr_review = []
    # filename = filename + ".txt"
    # f = open(filename, 'w', encoding='utf-8')
    for link in restaurant:
        # outstring = ''
        html = urllib.request.urlopen(link).read()
        soup = BeautifulSoup(html)

        store_name = soup.find_all('h1', class_='restaurant_name')
        name = store_name[0].get_text().strip()

        store_loc = soup.find_all('span', class_='orange-underline')
        loc = store_loc[0].get_text().strip()

        store_rate = soup.find_all('span', class_='rate-point')
        rate = store_rate[0].get_text().strip()

        
        data = soup.find_all('script', class_='review_menu_pictures_json', type='application/json')
        
        if len(data) > 0:
            for item in data:
                user_data = json.loads(item.text)
                user_id = user_data[0]['user']['member_uuid']
                user_review = user_data[0]['review']['comment']
                user_time = user_data[0]['review']['reg_time']
                if user_review not in usr_review:
                    str_name.append(name)
                    str_loc.append(loc)
                    str_rate.append(rate)
                    usr_id.append(user_id)
                    usr_time.append(user_time)
                    usr_review.append(user_review)


    dataset = pd.DataFrame({'store_name': str_name,
                            'store_location': str_loc,
                            'store_rating': str_rate,
                            'user_id': usr_id,
                            'user_time': usr_time,
                            'usr_review': usr_review})
    

    dataset.to_csv('test.csv', sep='\t', index=False)

        # reviewItem = soup.find_all('li', class_='default_review')

        # for review in reviewItems:
        #     # temp_review = review.find_all('span', class_=['short_review', 'more_review_bind', 'review_content'])
        #     # temp_date = review.find_all('span', class_=['past-time','ng-binding'])
        #     temp_rating = review.find_all('span', class_='icon-rating')

        #     if len(temp_review) > 0:
        #         # review_text = temp_review[0].get_text().strip()
        #         # review_date = temp_date[0]['ng-bind'].replace("from_date('",'')[:-2]
        #         user_rating = temp_rating[0].get_text().strip()
        # outstring = name + '\t' + loc + '\t' + str(rate) + '\t' + str(user_id) + '\t' + user_time + '\t' + user_review.replace('\n', '$e$') + '\n'
        # print(outstring)
    #     f.write(outstring)
        
    # f.close()


def main():
    restaurant = get_store()
    makingDataset(restaurant)

if __name__ == "__main__":
    # filename = sys.argv[1]
    main()
