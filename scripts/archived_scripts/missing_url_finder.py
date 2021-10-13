import json
from tqdm import tqdm
import os
import collections

DIR_PATH = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/'

with open(DIR_PATH + "newspaper-navigator-master/beyond_words_data/trainval.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

urls, file_names = [], []
for i, json in enumerate(tqdm(jsonObject['images'])):
    path, dirs, files = next(os.walk(DIR_PATH + 'scraped_photos/'))

    url_split = json['url'].split('/')
    file_name = url_split[-4]+'_'+url_split[-3]+'_'+url_split[-2]+'_'+url_split[-1]

    urls.append(json['url'])
    file_names.append(file_name)
    if file_name not in files:
        print(f'Missing url number: {i}')
        print(file_name)
        print(json['url'])
        print(' ')

# in my case upper condition was not activated at all
# some files could be duplicated it would solve mystery of missing photos (in my case 3)
print([item for item, count in collections.Counter(urls).items() if count > 1])
print([item for item, count in collections.Counter(file_names).items() if count > 1])
