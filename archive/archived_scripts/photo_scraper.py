import requests
import json
from tqdm import tqdm
import os

DIR_PATH = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/'

with open(DIR_PATH + "newspaper-navigator-master/beyond_words_data/trainval.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

for i, json in enumerate(tqdm(jsonObject['images'])):
    response = requests.get(json['url'])

    url_split = json['url'].split('/')
    file_name = url_split[-4]+'_'+url_split[-3]+'_'+url_split[-2]+'_'+url_split[-1]
    output_path = DIR_PATH + 'scraped_photos/' + file_name

    file = open(output_path, "wb")
    file.write(response.content)
    file.close()

    path, dirs, files = next(os.walk(DIR_PATH + 'scraped_photos/'))

    if (len(files)-1) != (i+1):
        print(output_path)
        print(len(files)-1)
        print(i)
        break