import json
import logging
from fastmcp import FastMCP
from sqlalchemy import select, func 
from db.database import AsyncSessionLocal, engine, Base
from db.models import Device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("postgres_mcp_server")

# 1. Initialize FastMCP Server
mcp = FastMCP("HexPilot Device Inventory MCP Server")

async def init_db_if_needed():
    """Ensure database tables exist in Neon PostgreSQL."""
    if engine:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


# 2. Expose MCP Resource: Schema Definition

@mcp.resource("postgres://devices/schema")
def get_device_schema() -> str:
    """Returns the SQL schema definition for the devices table."""
    return json.dumps(
        {
            "table": "devices",
            "columns": {
                "device_id": "VARCHAR(50) - Primary Key",
                "device_name": "VARCHAR(100) - Device hostname/name (Indexed)",
                "user_email": "VARCHAR(100) - Assigned owner email",
                "os_type": "VARCHAR(50) - Operating System (Windows, macOS, iOS, Android)",
                "os_version": "VARCHAR(50) - OS Build version",
                "serial_number": "VARCHAR(100) - Unique Hardware Serial",
                "compliance_status": "VARCHAR(50) - Compliant or Non-Compliant",
                "last_sync": "TIMESTAMP - Last telemetry sync date",
            },
        },
        indent=2,
    )


# 3. Expose MCP Tool 1: Lookup Device by Name
@mcp.tool()
async def get_device_by_name(device_name: str) -> str:
    """Fetch complete specifications and compliance status for a specific device name."""
    if not AsyncSessionLocal:
        return "Database connection not configured."

    await init_db_if_needed()

    async with AsyncSessionLocal() as session:
        stmt = select(Device).where(func.lower(Device.device_name) == device_name.lower())
        result = await session.execute(stmt)
        device = result.scalar_one_or_none()

        if not device:
            return f"Device '{device_name}' not found in inventory."

        return json.dumps(
            {
                "device_id": device.device_id,
                "device_name": device.device_name,
                "user_email": device.user_email,
                "os_type": device.os_type,
                "os_version": device.os_version,
                "serial_number": device.serial_number,
                "compliance_status": device.compliance_status,
                "last_sync": device.last_sync.isoformat() if device.last_sync else None,
            },
            indent=2,
        )


# 4. Expose MCP Tool 2: Search/Filter Devices
@mcp.tool()
async def search_devices(
    user_email: str = None, os_type: str = None, compliance_status: str = None
) -> str:
    """Search and filter devices by assigned user email, OS type (Windows/macOS/iOS/Android), or compliance status."""
    if not AsyncSessionLocal:
        return "Database connection not configured."

    await init_db_if_needed()

    async with AsyncSessionLocal() as session:
        query = select(Device)
        if user_email:
            query = query.where(func.lower(Device.user_email) == user_email.lower())
        if os_type:
            query = query.where(func.lower(Device.os_type) == os_type.lower())
        if compliance_status:
            query = query.where(func.lower(Device.compliance_status) == compliance_status.lower())

        result = await session.execute(query)
        devices = result.scalars().all()

        if not devices:
            return "No matching devices found."

        data = [
            {
                "device_id": d.device_id,
                "device_name": d.device_name,
                "user_email": d.user_email,
                "os_type": d.os_type,
                "compliance_status": d.compliance_status,
            }
            for d in devices
        ]
        return json.dumps(data, indent=2)


# 5. Expose MCP Tool 3: Get Compliance Summary Metrics
@mcp.tool()
async def get_compliance_summary() -> str:
    """Returns aggregated count metrics for total, compliant, and non-compliant devices in inventory."""
    if not AsyncSessionLocal:
        return "Database connection not configured."

    await init_db_if_needed()

    async with AsyncSessionLocal() as session:
        total_stmt = select(func.count(Device.device_id))
        total_res = await session.execute(total_stmt)
        total_count = total_res.scalar() or 0


        compliant_stmt = select(func.count(Device.device_id)).where(
            Device.compliance_status == "Compliant"
        )
        compliant_res = await session.execute(compliant_stmt)
        compliant_count = compliant_res.scalar() or 0

        non_compliant_count = total_count - compliant_count

        return json.dumps(
            {
                "total_devices": total_count,
                "compliant_devices": compliant_count,
                "non_compliant_devices": non_compliant_count,
                "compliance_rate": (
                    f"{(compliant_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
                ),
            },
            indent=2,
        )


if __name__ == "__main__":
    mcp.run()
