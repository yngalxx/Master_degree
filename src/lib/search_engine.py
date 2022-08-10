import os
import shutil

import pandas as pd
from tqdm import tqdm
from whoosh import fields, index, qparser

from lib.database import create_db_connection

# TODO: wrap it in a functions and speed it up (keywords part)

main_dir = (
    "/Users/alexdrozdz/Desktop/Studia/00. Seminarium"
    " magisterskie/Master_degree"
)
database_dir = "ocr_database"

conn = create_db_connection(f"{main_dir}/{database_dir}/newspapers_ocr.db")

temp_index_dir = ".temp"
if not os.path.exists(f"{main_dir}/{temp_index_dir}"):
    os.mkdir(f"{main_dir}/{temp_index_dir}")

df_ocr = pd.read_sql_query(
    "SELECT CLEANED_TEXT, PRED_LABEL, FILE_NAME FROM OCR_RESULTS", conn
)

df_ocr["SEARCH_TEXT"] = (
    df_ocr["PRED_LABEL"].str.lower() + " " + df_ocr["CLEANED_TEXT"].str.lower()
)

df_keywords = pd.read_sql_query(
    "SELECT KEYWORD, FILE_NAME FROM KEYWORDS", conn
)

df_keywords_flat = pd.DataFrame()

unique_files = df_keywords["FILE_NAME"].unique()

# TODO: speed it up (~40 sec), it has to be done in a second (maybe pandas has sth build in?) - if not before qui appears it should me small window with info that database is building
grouped_keywords_list = [
    " ".join(
        df_keywords["KEYWORD"][df_keywords["FILE_NAME"] == unique_file].values
    )
    for unique_file in tqdm(unique_files)
]
df_keywords_flat["KEYWORDS"] = grouped_keywords_list
df_keywords_flat["FILE_NAME"] = unique_files

df_ocr = df_ocr.merge(df_keywords_flat, on="FILE_NAME")

schema = fields.Schema(
    filename=fields.TEXT(stored=True),
    cleantext=fields.TEXT(stored=True),
    keywords=fields.TEXT(stored=True),
)

ix = index.create_in(f"{main_dir}/{temp_index_dir}", schema)

writer = ix.writer()
for i in tqdm(range(len(df_ocr)), desc="Indexing documents"):
    writer.add_document(
        filename=df_ocr["FILE_NAME"][i],
        cleantext=df_ocr["SEARCH_TEXT"][i],
        keywords=df_ocr["KEYWORDS"][i],
    )
writer.commit()

cleantext_list, file_name_list, keywords_list = [], [], []
with ix.searcher() as searcher:
    query = qparser.QueryParser("cleantext", ix.schema).parse(
        "sausage".lower()
    )
    results = searcher.search(query, terms=True)
    for r in results:
        cleantext_list.append(r["cleantext"])
        file_name_list.append(r["filename"])
        keywords_list.append(r["keywords"])

results_df = pd.DataFrame()
results_df["cleantext"] = cleantext_list
results_df["file_name"] = file_name_list
results_df["keywords"] = keywords_list
results_df["label"] = [x.split(maxsplit=1)[0] for x in results_df["cleantext"]]
results_df["cleantext"] = [
    x.split(maxsplit=1)[1] for x in results_df["cleantext"]
]

print(results_df)

shutil.rmtree(f"{main_dir}/{temp_index_dir}")
