import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

load_dotenv()

Base = declarative_base()

# Convert URL to asyncpg format cleanly (asyncpg expects 'ssl=' instead of 'sslmode=')
raw_url = os.getenv("DATABASE_URL", "")
if raw_url:
    url = (
        raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
        .replace("postgresql://", "postgresql+asyncpg://", 1)
        .replace("sslmode=require", "ssl=require")
        .replace("sslmode=prefer", "ssl=prefer")
        .replace("&channel_binding=require", "")
        .replace("channel_binding=require", "")
    )
    DATABASE_URL = url
else:
    DATABASE_URL = ""

# Initialize engine & session factory
engine = (
    create_async_engine(DATABASE_URL, pool_pre_ping=True, poolclass=NullPool)
    if DATABASE_URL
    else None
)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False) if engine else None

async def get_db_session():
    if not AsyncSessionLocal:
        raise RuntimeError("DATABASE_URL missing in .env")
    async with AsyncSessionLocal() as session:
        yield session
