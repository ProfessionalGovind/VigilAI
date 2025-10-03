from sqlmodel import create_engine, Session, SQLModel
from typing import Generator
import os

# DATABASE CONFIGURATION 
SQLITE_FILE_NAME = "vigilai_analytics.db" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_DB_URL = f"sqlite:///{BASE_DIR}/{SQLITE_FILE_NAME}"

# This engine handles the connection. Setting echo=True shows the raw SQL.
engine = create_engine(SQL_DB_URL, echo=True)

def create_db_and_tables():
    """
    This function runs one time to create the database file and all the tables 
    defined in our models.
    """
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """
    This is a special function (a dependency) that FastAPI uses. 
    It provides a database session for each request and closes it automatically.
    """
    with Session(engine) as session:
        yield session