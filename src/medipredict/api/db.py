import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from medipredict.config import BASE_DIR

DB_PATH = BASE_DIR / "history.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    condition = Column(String(50), nullable=False)
    input_data = Column(Text, nullable=False)
    probability = Column(Float, nullable=False)
    risk_label = Column(String(20), nullable=False)
    model_used = Column(String(100), nullable=False)
    timestamp = Column(String(50), nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
