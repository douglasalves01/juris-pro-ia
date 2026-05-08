"""Engine SQLAlchemy síncrono (Alembic, seed, pgvector)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import get_database_url_sync

engine = create_engine(get_database_url_sync(), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_sync_session() -> Session:
    return SessionLocal()
