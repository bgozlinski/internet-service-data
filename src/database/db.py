import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL = 'sqlite:///database.db'


def get_engine(database_url=None) -> sqlalchemy.engine:
    """
    Returns a SQLAlchemy engine instance.

    This function creates a new SQLAlchemy engine using the provided database URL.
    If no URL is provided, it defaults to the URL specified in the SQLALCHEMY_DATABASE_URL variable.

    Args:
        database_url (str, optional): The database URL to use for creating the engine. Defaults to None.

    Returns:
        sqlalchemy.engine.Engine: The SQLAlchemy engine instance.
    """
    if database_url is None:
        database_url = SQLALCHEMY_DATABASE_URL
    return create_engine(database_url)


engine = get_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Generator function that provides a database session.

    This function is intended for use with dependency injection in web applications,
    ensuring that a new database session is created for each request and properly closed after use.

    Yields:
        SessionLocal: An instance of a SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

