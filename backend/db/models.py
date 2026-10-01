from sqlalchemy import Column, String, DateTime, func
from db.database import Base

class Device(Base):
    __tablename__ = "devices"

    device_id = Column(String(50), primary_key=True)
    device_name = Column(String(100), nullable=False, index=True)
    user_email = Column(String(100), index=True)
    os_type = Column(String(50))         # e.g. "Windows", "macOS", "iOS", "Android"
    os_version = Column(String(50))      # e.g. "Windows 11 23H2", "macOS Sequoia"
    serial_number = Column(String(100), unique=True)
    compliance_status = Column(String(50), default="Compliant")  # "Compliant" or "Non-Compliant"
    last_sync = Column(DateTime, default=func.now(), onupdate=func.now())
