import logging
import os
import sqlite3
import sys

import click
import cv2
import keybert
import pytesseract
import spacy
from constants import General, Output
from tqdm import tqdm

from lib.database import create_db_connection, create_db_tables, db_count, db_insert, db_drop
from lib.logs import Log
from lib.ocr import (combine_data_for_ocr, crop_image, get_keywords,
                     image_transform, ocr_init, ocr_predict, ocr_text_clean)
from lib.save_load_data import from_tsv_to_list


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    type=str,
    default=General.MAIN_DIR,
    help="Path to the level where the repository is stored.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    default=Output.MIN_CONF_LEVEL,
    type=float,
    help="Minimum confidence level for model predictions to show up.",
    show_default=True,
)
@click.option(
    "--update_existing",
    default=Output.UPDATE_EXISTING_OCR,
    type=bool,
    help="If False: crop visual content from origin files, else: read already collected visual content.",
    show_default=True,
)
def ocr_runner(main_dir, min_conf_level, update_existing):
    # initialize logger
    logger = Log("ocr_runner", main_dir)
    logger.log_start()

    # check provided path
    test_dir = "data/test-A"
    assert os.path.exists(f"{main_dir}/{test_dir}") == True

    try:
        in_file_name = "in.tsv"
        in_list = from_tsv_to_list(
            path=f"{main_dir}/{test_dir}/{in_file_name}"
        )
    except FileNotFoundError as err:
        logging.error(
            f"File '{in_file_name}' not found, code will be forced to"
            f" quit...\nError: {err}"
        )
        sys.exit(1)
    try:
        out_file_name = "out.tsv"
        out_list = from_tsv_to_list(
            path=f"{main_dir}/{test_dir}/{out_file_name}"
        )
    except FileNotFoundError as err:
        logging.error(
            f"File '{out_file_name}' not found, code will be forced to"
            f" quit...\nError: {err}"
        )
        sys.exit(1)

    logging.info("Combining in and out test files for OCR")
    innout = combine_data_for_ocr(in_list, out_list, min_conf_level)
    logging.info(
        f"OCR input size: {len(innout)}, declared minimum confidence level:"
        f" {min_conf_level}"
    )

    logging.info("Initializing Tesseract OCR")
    try:
        pytesseract.pytesseract.tesseract_cmd = ocr_init()
    except ModuleNotFoundError as err:
        logging.error(
            "Tesseract OCR not found, try: 'brew install tesseract'\nError:"
            f" {err}"
        )
        sys.exit(1)

    vc_content_dir = "cropped_visual_content"
    if not os.path.exists(f"{main_dir}/{vc_content_dir}"):
        logging.info(
            f"Directory '{vc_content_dir}' doesn't exist, creating one"
        )
        os.makedirs(f"{main_dir}/{vc_content_dir}")
        update_existing = False

    # spacy language core
    logging.info("Loading spaCy language core")
    try:
        nlp = spacy.load("en_core_web_sm")
    except ModuleNotFoundError as err:
        logging.error(
            "Language core not found, try: 'python -m spacy download"
            f" en_core_web_sm'\nError: {err}"
        )
        sys.exit(1)

    # keybert model to extract keywords
    logging.info("Loading KeyBERT pretrained model")
    kw_model = keybert.KeyBERT(model="all-mpnet-base-v2")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # database for ocr results
    database_dir = "ocr_database"
    if not os.path.exists(f"{main_dir}/{database_dir}"):
        logging.info(f"Directory '{database_dir}' doesn't exist, creating one")
        os.makedirs(f"{main_dir}/{database_dir}")

    logging.info("Creating database instance")
    try:
        conn = create_db_connection(
            f"{main_dir}/{database_dir}/newspapers_ocr.db"
        )
    except sqlite3.Error as err:
        logging.error(
            "Cannot create database instance, code will be forced to"
            f" quit...\nError: {err}"
        )
        sys.exit(1)

    if update_existing:
        logging.info(f'Deleting existing tables, caused by setting update_existing argument to {update_existing}')
        db_drop(conn, 'OCR_RESULTS')
        db_drop(conn, 'KEYWORDS')

    logging.info("Creating tables OCR_RESULTS and KEYWORDS")
    try:
        create_db_tables(conn)
    except sqlite3.Error as err:
        logging.error(
            "Cannot create tables, code will be forced to quit...\nError:"
            f" {err}"
        )
        sys.exit(1)

    logging.info(
        "Processing images, applying OCR, cleaning text, extracting keywords and saving metadata"
    )
    keyword_iterator = 1
    for i, elem in enumerate(tqdm(innout, desc="OCR running")):
        cropped_img_name = f'vc_{i+1}.png'
        if not update_existing:
            # read image
            img = cv2.imread(f"{main_dir}/scraped_photos/{elem[0]}")
            # crop visual content
            cropped_img = crop_image(image=img, x0=elem[2], x1=elem[4], y0=elem[3], y1=elem[5])
        else:
            cropped_img = cv2.imread(f"{main_dir}/{vc_content_dir}/{cropped_img_name}")
        # transform visual content
        transformed_cropped_img = image_transform(cropped_img)
        # ocr
        cropped_img_str = ocr_predict(transformed_cropped_img)
        # clean text
        clean_txt, normalized_txt = ocr_text_clean(
            cropped_img_str, spacy_language_core=nlp
        )
        # keywords
        keywords_list = get_keywords(
            clean_txt,
            top_n=10,
            keybert_model=kw_model,
            ngram=1,
            only_this_ngram=True,
            language="english",
        )
        # save results
        if not update_existing:
            cv2.imwrite(
                f"{main_dir}/{vc_content_dir}/{cropped_img_name}",
                cropped_img,
            )
        db_insert(
            conn,
            "OCR_RESULTS",
            (
                i + 1,
                elem[0],
                cropped_img_name,
                elem[1],
                cropped_img_str,
                clean_txt.strip(),
                " ".join(normalized_txt),
            ),
        )

        for keyword in keywords_list:
            db_insert(
                conn,
                "KEYWORDS",
                (
                    keyword_iterator,
                    keyword["keyword"],
                    keyword["score"],
                    cropped_img_name,
                ),
            )
            keyword_iterator += 1

    logging.info(
        f"All {i+1} visual content images were successfully stored in '{vc_content_dir}' directory"
    )
    logging.info(f"Table OCR_RESULTS count: {db_count(conn, 'OCR_RESULTS')}")
    logging.info(f"Table KEYWORDS count: {db_count(conn, 'KEYWORDS')}")

    # end logger
    logger.log_end()


if __name__ == "__main__":
    ocr_runner()
