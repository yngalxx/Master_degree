import os
import re
import warnings
from typing import List, Tuple

import cv2
import keybert
import numpy as np
import pytesseract
import spacy

# warnings
warnings.filterwarnings("ignore")


def combine_data_for_ocr(
    in_list: List, out_list: List, confidence_level: float
) -> List:
    innout = []
    for i in range(len(out_list)):
        temp_out_list = out_list[i].split(" ")
        for annotation in temp_out_list:
            temp_annotation = annotation.split(":")
            if float(temp_annotation[2]) >= confidence_level:
                bbox_list = temp_annotation[1].split(",")
            else:
                continue
            innout.append(
                [
                    in_list[i],
                    temp_annotation[0],
                    int(bbox_list[0]),
                    int(bbox_list[1]),
                    int(bbox_list[2]),
                    int(bbox_list[3]),
                ]
            )

    return innout


def ocr_init() -> str:
    return os.popen("brew list tesseract | grep 'bin'").read().strip()


def crop_image(
    image: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> np.ndarray:
    """
    Crop image using bboxes
    """
    return image[y0:y1, x0:x1]


def image_transform(image: np.ndarray) -> np.ndarray:
    """
    Image transformation pipline
    """
    # greyscale image
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # dilate the image to get background (text removal)
    dilated_img = cv2.dilate(grey_img, np.ones((7, 7), np.uint8))
    # use median blur on dilated image to get better background image containing all the shadows and discoloration
    bg_img = cv2.medianBlur(dilated_img, 21)
    # combine new backgorund with old image
    diff_img = 255 - cv2.absdiff(grey_img, bg_img)
    # normalize the image to get full dynamic range
    norm_img = cv2.normalize(
        diff_img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )
    return norm_img


def ocr_predict(image: np.ndarray) -> str:
    """
    Read text from image
    """
    return pytesseract.image_to_string(image)


def ocr_text_clean(
    text: str, spacy_language_core: spacy.language.Language
) -> Tuple:
    # regex to clean and prepare text for search
    re_clean = re.compile("[^a-zA-Z1-9\s,.!?$%-']")
    re_search = re.compile("[^a-zA-Z1-9\s-]")
    # line-breaks fix
    fixed_text = re.sub("\n", " ", re.sub("-\n", "", text))
    # clean text
    clean_txt = re_clean.sub("", fixed_text)
    clean_txt = re.sub(" +", " ", clean_txt)
    # normalized text
    normalized_txt = re_search.sub("", fixed_text)
    normalized_txt = re.sub("-", " ", normalized_txt)
    normalized_txt = [
        token.lemma_.lower()
        for token in spacy_language_core(normalized_txt)
        if not token.is_stop and not token.is_punct
    ]
    normalized_txt = [word for word in normalized_txt if len(word) > 1]
    return clean_txt, normalized_txt


def get_keywords(
    ocr_text: str,
    keybert_model: keybert.KeyBERT,
    top_n: int = 20,
    ngram: int = 1,
    only_this_ngram: bool = True,
    language: str = "english",
) -> List:
    keywords = keybert_model.extract_keywords(
        ocr_text,
        keyphrase_ngram_range=(1, ngram),
        stop_words=language,
        highlight=False,
        top_n=top_n,
    )
    if only_this_ngram:
        keywords = [
            keyword
            for keyword in keywords
            if len(keyword[0].split(" ")) == ngram
        ]

    return [
        {"keyword": keyword[0], "score": keyword[1]} for keyword in keywords
    ]
