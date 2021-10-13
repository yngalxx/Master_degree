import requests
import json
from tqdm import tqdm
import os

DIR_PATH = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/'

with open(DIR_PATH + "Master_degree/additional_data/bw_website_data.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

url_list = []
for i in range(len(jsonObject['data'])):
    url_list.append(jsonObject['data'][i]['location']['standard'])

url_list_no_duplicates = list(dict.fromkeys(url_list))

for url in tqdm(url_list_no_duplicates):
    path, dirs, files = next(os.walk(DIR_PATH + 'scraped_photos_more_data/'))

    url_split = url.split('/')
    file_name = url_split[-4] + '_' + url_split[-3] + '_' + url_split[-2] + '_' + url_split[-1]

    if file_name not in files:
        response = requests.get(url)
        output_path = DIR_PATH + 'scraped_photos_more_data/' + file_name
        file = open(output_path, "wb")
        file.write(response.content)
        file.close()
