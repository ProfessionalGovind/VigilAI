from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel # We don't need create_engine here, but keep Field and SQLModel

# DATABASE MODEL: This is the blueprint for our PostgreSQL table
class VigilEvent(SQLModel, table=True):
    """
    This is the blueprint for the table in my database. It tracks every 
    detection event the system generates.
    """
    # Auto-incrementing ID for the primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # 1. Event Context
    model_used: str = Field(index=True) # Tells me if YOLO or FRCNN was used
    timestamp: datetime = Field(default_factory=datetime.utcnow) # When the event happened
    
    # 2. Analytics Data
    total_detections: int = Field(default=0) # Total objects seen in the frame
    person_count: int = Field(default=0)
    
    # 3. Anomaly Flag (Our extended goal)
    is_anomaly: bool = Field(default=False) # Flag if the density was too high
    
    # 4. Location/Metadata
    video_source: str # Which video file/camera the event came from