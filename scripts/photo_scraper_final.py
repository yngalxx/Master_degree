import requests
import json
from tqdm import tqdm
import os

DIR_PATH = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/'

# getting photos in better resolution from urls finded in COCO formatted
# files obtained from source repository (newspaper-navigator-master)
with open(DIR_PATH + "Master_degree/additional_data/trainval.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

for i in tqdm(range(len(jsonObject['images']))):
    output_path = DIR_PATH + 'scraped_photos_final/'
    path, dirs, files = next(os.walk(output_path))

    file_name = jsonObject['images'][i]['file_name']

    if file_name not in files:
        response = requests.get(jsonObject['images'][i]['url'])
        file = open(output_path + file_name, "wb")
        file.write(response.content)
        file.close()
        