import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from db import DB


async def run():
    cfg = load_config()
    db = DB(cfg.db_path)
    await db.connect()

    for chat_id in cfg.allowed_chat_ids:
        await db.ensure_allowed_chat(chat_id, title="(from env)")

    await db.close()
    print("DB initialized ok")


if __name__ == "__main__":
    asyncio.run(run())
