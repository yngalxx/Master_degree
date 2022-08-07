import json
import logging
import os

import click
import requests
from constants import General
from tqdm import tqdm

from lib.logs import Log


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Path to the level where this repository is stored.",
    show_default=True,
)
def image_scraper(main_dir):
    # initialize logger
    logger = Log("scraper_runner", main_dir)
    logger.log_start()

    # check provided path
    source_annotations_dir = "source_annotations"
    assert os.path.exists(f"{main_dir}/{source_annotations_dir}") == True

    # check whether the scraped_photos directory exists, and if not create it
    final_dir = "scraped_photos/"
    final_path = f"{main_dir}/{final_dir}"
    if not os.path.exists(final_path):
        logging.info(f'Directory "{final_dir}" doesn\'t exist, creating one')
        os.makedirs(final_path)

    # read source annotations file
    try:
        annotations_source_file = "trainval.json"
        with open(
            f"{main_dir}/{source_annotations_dir}/{annotations_source_file}"
        ) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
    except:
        logging.error(
            f"File '{annotations_source_file}' not found, code will be forced"
            " to quit"
        )
        raise FileNotFoundError()

    images_num = len(jsonObject["images"])
    logging.info(f"Collecting {images_num} images")
    for i in tqdm(range(images_num), desc="Scraping"):
        _, _, files = next(os.walk(final_path))

        file_name = jsonObject["images"][i]["file_name"]

        if file_name not in files:
            response = requests.get(jsonObject["images"][i]["url"])
            with open(final_path + file_name, "wb") as file:
                file.write(response.content)

    # end logger
    logger.log_end()


if __name__ == "__main__":
    image_scraper()
