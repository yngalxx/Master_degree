import logging
import click
from lib.ocr import combine_data_for_ocr, ocr_init, crop_image, image_transform, ocr_predict, ocr_text_clean, get_keywords
from lib.logs import Log
from constants import General, Output
import os
import pytesseract
from tqdm import tqdm
import cv2
import spacy
import keybert
from lib.save_load_data import from_tsv_to_list, dump_json


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    type=str,
    default=General.MAIN_DIR,
    help="Path to the level where this repository is stored.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    default=Output.MIN_CONF_LEVEL,
    type=float,
    help="Minimum confidence level for model predictions to show up.",
    show_default=True,
)
def ocr_runner(main_dir, min_conf_level):
    # initialize logger
    logger = Log("ocr_runner", main_dir)
    logger.log_start()

    # check provided path
    test_dir = "data/test-A"
    assert os.path.exists(f"{main_dir}/{test_dir}") == True

    try:
        in_file_name = "in.tsv"
        in_list = from_tsv_to_list(path=f"{main_dir}/{test_dir}/{in_file_name}")
    except:
        logging.error(
            f"File '{in_file_name}' not found, code will be forced to quit"
        )
        raise FileNotFoundError()
    try:
        out_file_name = "out.tsv"
        out_list = from_tsv_to_list(path=f"{main_dir}/{test_dir}/{out_file_name}")
    except:
        logging.error(
            f"File '{out_file_name}' not found, code will be forced to quit"
        )
        raise FileNotFoundError()

    logging.info("Combining in and out test files for OCR")
    innout = combine_data_for_ocr(in_list, out_list, min_conf_level)
    
    logging.info("Initializing Tesseract OCR")
    try:
        pytesseract.pytesseract.tesseract_cmd = ocr_init()
    except:
        logging.error('Tesseract OCR not found, check if you installed it correctly')
        raise ModuleNotFoundError()

    vs_content_dir = "visual_content"
    if not os.path.exists(f"{main_dir}/{vs_content_dir}"):
        logging.info(f"Directory '{vs_content_dir}' doesn't exist, creating one")
        os.makedirs(f"{main_dir}/{vs_content_dir}")

    # spacy language core
    logging.info("Loading Spacy language core")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        logging.error('Spacy language core not found, check if you installed it correctly')
        raise ModuleNotFoundError()

    # keybert model to extract keywords
    logging.info("Loading KeyBERT pretrained model")
    kw_model = keybert.KeyBERT(model='all-mpnet-base-v2')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.info("Cropping visual contents, transforming, using OCR, cleaning results and saving")
    final_dict = {}
    for i, elem in enumerate(tqdm(innout, desc='OCR running')):
        # read image
        img = cv2.imread(f'{main_dir}/scraped_photos/{elem[0]}')
        # crop visual content
        cropped_img = crop_image(img, elem[2], elem[4], elem[3], elem[5])
        # transform visual content
        transformed_cropped_img = image_transform(cropped_img)
        # ocr 
        cropped_img_str = ocr_predict(transformed_cropped_img)
        # clean text
        clean_txt, search_txt = ocr_text_clean(cropped_img_str, spacy_language_core=nlp)
        # keywords
        keywords_list = get_keywords(search_txt, top_n=10, keybert_model=kw_model)
        # save results
        in_dict = {'origin_file': elem[0], 'predicted_label': elem[1], 'ocr_raw_text': cropped_img_str, 'cleaned_text': clean_txt.strip(), 'search_text': search_txt.strip(), 'keywords': keywords_list}
        cv2.imwrite(f"{main_dir}/visual_content/vs_{i}.png", cropped_img)
        final_dict[f"vs_{i}.png"] = in_dict

    ocr_dir = "ocr_results"
    if not os.path.exists(f"{main_dir}/ocr_results"):
        logging.info(f"Directory '{ocr_dir}' doesn't exist, creating one")
        os.makedirs(f"{main_dir}/ocr_results")

    dump_json(path=f'{ocr_dir}/vs_ocr_data.json', dict_to_save=final_dict)
    logging.info(f"OCR output json saved in '{ocr_dir}' directory")

    # end logger
    logger.log_end()


if __name__ == "__main__":
    ocr_runner()
