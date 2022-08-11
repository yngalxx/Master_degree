import os
import shutil

import pandas as pd
import whoosh

from lib.database import create_db_connection


def prepare_data(main_dir: str) -> pd.DataFrame:
    """
    Prepare input data for indexing before using search engine
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

    return df_ocr


def create_temp_dir(main_dir: str) -> None:
    """
    Create temporary directory for search engine purposes
    """
    os.mkdir(f"{main_dir}/.temp")


def index_documents(main_dir: str, df_ocr: pd.DataFrame) -> None:
    """
    Prepare input data for full text searching by indexing documents
    """
    schema = whoosh.fields.Schema(
        filename=whoosh.fields.TEXT(stored=True),
        cleantext=whoosh.fields.TEXT(stored=True),
        keywords=whoosh.fields.TEXT(stored=True),
    )
    ix = whoosh.index.create_in(f"{main_dir}/.temp", schema)
    writer = ix.writer()
    for i in range(len(df_ocr)):
        writer.add_document(
            filename=df_ocr["FILE_NAME"][i],
            cleantext=df_ocr["SEARCH_TEXT"][i],
        )
    writer.commit()

    return ix


def full_text_search(ix: whoosh.FileIndex, query: str) -> pd.DataFrame:
    """
    Full text search through indexed documents
    """
    file_name_list = []
    with ix.searcher() as searcher:
        query = whoosh.qparser.QueryParser("cleantext", ix.schema).parse(
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
    shutil.rmtree(f"{main_dir}/.temp")
