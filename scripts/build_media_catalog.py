import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiogram import Bot
from dotenv import load_dotenv

from media import build_media_catalog, save_media_catalog


async def main():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    load_dotenv(ROOT / ".env")
    token = (os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN missing")

    out_path = (os.getenv("MEDIA_CATALOG_PATH") or "./media_catalog.json").strip()
    bot = Bot(token)
    try:
        catalog = await build_media_catalog(bot)
    finally:
        await bot.session.close()

    save_media_catalog(str((ROOT / out_path).resolve()), catalog)
    print(
        f"saved items={len(catalog.get('items', []))} "
        f"failed_sets={len(catalog.get('failed_sets', []))}"
    )


if __name__ == "__main__":
    asyncio.run(main())
