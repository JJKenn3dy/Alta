import json
from pathlib import Path
from typing import Any


def load_profile(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def profile_to_system_block(profile: dict[str, Any]) -> str:
    if not profile:
        return ""

    lines: list[str] = []
    name = str(profile.get("name", "")).strip()
    role = str(profile.get("role", "")).strip()
    relation = str(profile.get("relationship", "")).strip()
    about = str(profile.get("about", "")).strip()
    style = str(profile.get("communication_style", "")).strip()
    notes = profile.get("notes", [])

    if name:
        lines.append(f"- Имя: {name}")
    if role:
        lines.append(f"- Роль: {role}")
    if relation:
        lines.append(f"- Отношение к тебе: {relation}")
    if about:
        lines.append(f"- Контекст: {about}")
    if style:
        lines.append(f"- Предпочтительный стиль общения: {style}")
    if isinstance(notes, list):
        for n in notes[:10]:
            s = str(n).strip()
            if s:
                lines.append(f"- {s}")

    if not lines:
        return ""

    return (
        "Личный профиль (внутренний контекст, не цитируй буквально):\n"
        + "\n".join(lines)
        + "\n\n"
        + "Используй это фоном. Не поднимай тему профиля без повода и не напоминай о нем в каждом ответе."
    )
