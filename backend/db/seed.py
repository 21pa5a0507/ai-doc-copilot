import asyncio
import logging
from sqlalchemy import select
from db.database import engine, AsyncSessionLocal, Base
from db.models import Device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DEVICES = [
    {
        "device_id": "DEV-1001",
        "device_name": "WIN-DESKTOP-ALEX",
        "user_email": "alex.smith@hexnode.com",
        "os_type": "Windows",
        "os_version": "Windows 11 23H2",
        "serial_number": "SN-WIN-99812",
        "compliance_status": "Compliant",
    },
    {
        "device_id": "DEV-1002",
        "device_name": "MACBOOK-PRO-KISHORE",
        "user_email": "kishore@hexnode.com",
        "os_type": "macOS",
        "os_version": "macOS Sequoia 15.1",
        "serial_number": "SN-MAC-77341",
        "compliance_status": "Compliant",
    },
    {
        "device_id": "DEV-1003",
        "device_name": "IPHONE-CORP-09",
        "user_email": "sarah.connor@hexnode.com",
        "os_type": "iOS",
        "os_version": "iOS 17.5",
        "serial_number": "SN-IOS-12004",
        "compliance_status": "Non-Compliant",
    },
    {
        "device_id": "DEV-1004",
        "device_name": "WIN-LAPTOP-DEV04",
        "user_email": "alex.smith@hexnode.com",
        "os_type": "Windows",
        "os_version": "Windows 10 22H2",
        "serial_number": "SN-WIN-44102",
        "compliance_status": "Non-Compliant",
    },
    {
        "device_id": "DEV-1005",
        "device_name": "PIXEL-TEST-8A",
        "user_email": "dev.ops@hexnode.com",
        "os_type": "Android",
        "os_version": "Android 14",
        "serial_number": "SN-AND-88319",
        "compliance_status": "Compliant",
    },
]

async def seed_database():
    if not engine:
        logger.error("Database engine is not initialized. Check your DATABASE_URL in .env.")
        return

    logger.info("Connecting to Neon PostgreSQL and creating tables...")
    async with engine.begin() as conn:
        # Create all tables defined in Base (including 'devices')
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Tables created successfully!")

    async with AsyncSessionLocal() as session:
        # Check if sample devices already exist
        result = await session.execute(select(Device))
        existing_devices = result.scalars().all()

        if existing_devices:
            logger.info(f"Database already contains {len(existing_devices)} devices. Skipping seed.")
            return

        logger.info("Inserting sample device inventory into Neon PostgreSQL...")
        for dev_data in SAMPLE_DEVICES:
            device = Device(**dev_data)
            session.add(device)
        
        await session.commit()
        logger.info("Successfully seeded 5 sample devices into Neon PostgreSQL!")

if __name__ == "__main__":
    asyncio.run(seed_database())
