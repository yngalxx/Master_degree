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
    help="Path to the level where this repository is stored",
    show_default=True,
)
def image_scraper(directory_path):
    """
    Simple scraper to retriev full-resolution images from urls finded in COCO
    formatted files with annotations obtained from source repository (newspaper-navigator-master)
    """

    # check whether the scraped_photos directory exists, and if not create it
    final_dir = "scraped_photos/"
    final_path = f"{directory_path}/{final_dir}"
    if not os.path.exists(final_path):
        print("Directory 'scraped_photos' doesn't exist, creating one ...")
        os.makedirs(final_path)

    with open(
        f"{directory_path}/source_annotations/trainval.json"
    ) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    for i in tqdm(range(len(jsonObject["images"]))):
        _, _, files = next(os.walk(final_path))

        file_name = jsonObject["images"][i]["file_name"]

        if file_name not in files:
            response = requests.get(jsonObject["images"][i]["url"])
            with open(final_path + file_name, "wb") as file:
                file.write(response.content)


if __name__ == "__main__":
    image_scraper()
