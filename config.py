from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _parse_chat_ids(value: str) -> set[int]:
    value = (value or "").strip()
    if not value:
        return set()
    out: set[int] = set()
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.add(int(x))
        except ValueError:
            raise ValueError(f"ALLOWED_CHAT_IDS contains non-integer value: {x!r}")
    return out


def _parse_optional_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"expected integer, got {value!r}")


def _env_int(name: str, default: str, *, min_val: int | None = None, max_val: int | None = None) -> int:
    raw = os.getenv(name, default).strip()
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    if min_val is not None and val < min_val:
        raise ValueError(f"{name}={val} is below minimum {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"{name}={val} is above maximum {max_val}")
    return val


def _env_float(name: str, default: str, *, min_val: float | None = None, max_val: float | None = None) -> float:
    raw = os.getenv(name, default).strip()
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {raw!r}")
    if min_val is not None and val < min_val:
        raise ValueError(f"{name}={val} is below minimum {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"{name}={val} is above maximum {max_val}")
    return val


@dataclass(frozen=True)
class Config:
    bot_token: str
    allowed_chat_ids: set[int]
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_context_window: int
    llm_context_safety_margin: int
    db_path: str
    history_limit: int
    temperature: float
    top_p: float
    max_tokens: int
    user_cooldown_sec: int
    debug_traces: bool
    enable_debug_logs: bool
    enable_metrics: bool
    orchestration_log_path: str
    creator_user_id: int | None
    creator_profile_path: str
    join_verification_timeout_sec: int
    media_catalog_path: str
    media_reply_prob: float
    media_reply_cooldown_sec: int
    knowledge_dir: str
    knowledge_max_files: int
    knowledge_max_file_bytes: int


def load_config() -> Config:
    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is required")

    llm_context_window = _env_int("LLM_CONTEXT_WINDOW", "8000", min_val=512)
    llm_context_safety_margin = _env_int("LLM_CONTEXT_SAFETY_MARGIN", "512", min_val=0)
    max_tokens = _env_int("MAX_TOKENS", "90", min_val=1)

    if max_tokens >= llm_context_window:
        raise ValueError(
            f"MAX_TOKENS ({max_tokens}) must be less than LLM_CONTEXT_WINDOW ({llm_context_window})"
        )

    return Config(
        bot_token=bot_token,
        allowed_chat_ids=_parse_chat_ids(os.getenv("ALLOWED_CHAT_IDS", "")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "local-llm").strip(),
        llm_model=os.getenv(
            "LLM_MODEL",
            "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
        ).strip(),
        llm_context_window=llm_context_window,
        llm_context_safety_margin=llm_context_safety_margin,
        db_path=os.getenv("DB_PATH", "./alta_tg_test.db").strip(),
        history_limit=_env_int("HISTORY_LIMIT", "40", min_val=1),
        temperature=_env_float("TEMPERATURE", "0.55", min_val=0.0, max_val=2.0),
        top_p=_env_float("TOP_P", "0.85", min_val=0.0, max_val=1.0),
        max_tokens=max_tokens,
        user_cooldown_sec=_env_int("USER_COOLDOWN_SEC", "15", min_val=0),
        debug_traces=os.getenv("DEBUG_TRACES", "0").strip().lower() in {"1", "true", "yes", "on"},
        enable_debug_logs=os.getenv("ENABLE_DEBUG_LOGS", "1").strip().lower() in {"1", "true", "yes", "on"},
        enable_metrics=os.getenv("ENABLE_METRICS", "1").strip().lower() in {"1", "true", "yes", "on"},
        orchestration_log_path=os.getenv("ORCHESTRATION_LOG_PATH", "./orchestration.log").strip(),
        creator_user_id=_parse_optional_int(os.getenv("CREATOR_USER_ID", "")),
        creator_profile_path=os.getenv("CREATOR_PROFILE_PATH", "./creator_profile.json").strip(),
        join_verification_timeout_sec=_env_int("JOIN_VERIFICATION_TIMEOUT_SEC", "90", min_val=5, max_val=600),
        media_catalog_path=os.getenv("MEDIA_CATALOG_PATH", "./media_catalog.json").strip(),
        media_reply_prob=_env_float("MEDIA_REPLY_PROB", "0.22", min_val=0.0, max_val=1.0),
        media_reply_cooldown_sec=_env_int("MEDIA_REPLY_COOLDOWN_SEC", "120", min_val=0),
        knowledge_dir=os.getenv("KNOWLEDGE_DIR", "./knowledge").strip(),
        knowledge_max_files=_env_int("KNOWLEDGE_MAX_FILES", "64", min_val=0),
        knowledge_max_file_bytes=_env_int("KNOWLEDGE_MAX_FILE_BYTES", "262144", min_val=0),
    )
