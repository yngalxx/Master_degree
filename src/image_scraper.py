import json
import os
import pathlib

import click
import requests
from tqdm import tqdm


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--directory_path",
    default="/".join(
        str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1]
    ),
    type=str,
    help="Path to directory where this repository is stored",
    show_default=True,
)
def image_scraper(directory_path):
    """
    simple scraper to retriev full-resolution images from urls finded in COCO formatted files with annotations obtained from source repository (newspaper-navigator-master)
    """
    with open(directory_path + "/additional_data/trainval.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    for i in tqdm(range(len(jsonObject["images"]))):
        output_path = directory_path + "scraped_photos/"
        path, dirs, files = next(os.walk(output_path))

        file_name = jsonObject["images"][i]["file_name"]

        if file_name not in files:
            response = requests.get(jsonObject["images"][i]["url"])
            file = open(output_path + file_name, "wb")
            file.write(response.content)
            file.close()


if __name__ == "__main__":
    image_scraper()
