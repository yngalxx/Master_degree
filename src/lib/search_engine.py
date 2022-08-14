import os
import shutil
from typing import List

import pandas as pd
from whoosh import index, fields, qparser

from lib.database import create_db_connection


def prepare_data(main_dir: str) -> index.FileIndex:
    """
    Prepare input data and index documents
    """
    conn = create_db_connection(f"{main_dir}/ocr_database/newspapers_ocr.db")

    df_ocr = pd.read_sql_query(
        "SELECT CLEANED_TEXT, PRED_LABEL, FILE_NAME FROM OCR_RESULTS", conn
    )
    df_ocr["SEARCH_TEXT"] = (
        df_ocr["PRED_LABEL"].str.lower()
        + " "
        + df_ocr["CLEANED_TEXT"].str.lower()
    )

    os.mkdir(f"{main_dir}/.temp")

    schema = fields.Schema(
        filename=fields.TEXT(stored=True),
        cleantext=fields.TEXT(stored=True),
        keywords=fields.TEXT(stored=True),
    )
    ix = index.create_in(f"{main_dir}/.temp", schema)
    writer = ix.writer()
    for i in range(len(df_ocr)):
        writer.add_document(
            filename=df_ocr["FILE_NAME"][i],
            cleantext=df_ocr["SEARCH_TEXT"][i],
        )
    writer.commit()

    return ix


def full_text_search(ix: index.FileIndex, query: str) -> List:
    """
    Full text search through indexed documents
    """
    file_name_list = []
    with ix.searcher() as searcher:
        query = qparser.QueryParser("cleantext", ix.schema).parse(
            query.lower()
        )
        results = searcher.search(query, terms=True)
        for r in results:
            file_name_list.append(r["filename"])

    return file_name_list


def remove_temp_dir(main_dir: str) -> None:
    """
    Remove temporary directory
    """
    tmp_path = f"{main_dir}/.temp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
