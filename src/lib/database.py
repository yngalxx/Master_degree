import sqlite3
from typing import Tuple


def create_db_connection(db_file_path: str) -> sqlite3.Connection:
    """
    Create a database connection to the SQLite database specified by db_file
    """
    return sqlite3.connect(db_file_path)


def create_db_tables(db_connection: sqlite3.Connection) -> None:
    """
    Create database tables to store OCR data
    """
    db_connection.execute(
        """
        CREATE TABLE OCR_RESULTS(
            ID                        INT PRIMARY KEY                        NOT NULL,
            ORIGIN_FILE_NAME          TEXT                                   NOT NULL,
            FILE_NAME                 TEXT                                   NOT NULL,
            PRED_LABEL                CHAR(20)                               NOT NULL,
            OCR_RAW_TEXT              TEXT,
            CLEANED_TEXT              TEXT,
            NORMALIZED_TEXT           TEXT
        );
        """
    )
    db_connection.execute(
        """
        CREATE TABLE KEYWORDS(
            ID                         INT PRIMARY KEY                       NOT NULL,
            KEYWORD                    TEXT,
            SCORE                      REAL,
            FILE_NAME                  TEXT                                  NOT NULL,
            FOREIGN KEY(FILE_NAME)     REFERENCES OCR_RESULTS(FILE_NAME)
        );
        """
    )


def db_insert(
    db_connection: sqlite3.Connection, table: str, values: Tuple
) -> None:
    """
    Insert row into database table
    """
    quot_marks = " ".join(
        ["?," if i + 1 != len(values) else "?" for i in range(len(values))]
    )
    query = f"""
        INSERT INTO {table}
        VALUES({quot_marks})
        """
    cursor = db_connection.cursor()
    cursor.execute(query, values)
    db_connection.commit()


def db_count(db_connection: sqlite3.Connection, table: str) -> int:
    """
    Get number of rows stored in database table
    """
    cursor = db_connection.cursor()
    count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return count


def db_drop(db_connection: sqlite3.Connection, table: str) -> None:
    """
    Drop table from database 
    """
    query = f"""DROP TABLE IF EXISTS {table}"""
    cursor = db_connection.cursor()
    cursor.execute(query)
    db_connection.commit()