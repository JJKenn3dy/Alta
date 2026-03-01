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
        out.add(int(x))
    return out


def _parse_optional_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


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

    return Config(
        bot_token=bot_token,
        allowed_chat_ids=_parse_chat_ids(os.getenv("ALLOWED_CHAT_IDS", "")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "local-llm").strip(),
        llm_model=os.getenv("LLM_MODEL", "saiga_nemo_12b.Q8_0.gguf").strip(),
        llm_context_window=int(os.getenv("LLM_CONTEXT_WINDOW", "4096")),
        llm_context_safety_margin=int(os.getenv("LLM_CONTEXT_SAFETY_MARGIN", "512")),
        db_path=os.getenv("DB_PATH", "./alta_tg_test.db").strip(),
        history_limit=int(os.getenv("HISTORY_LIMIT", "40")),
        temperature=float(os.getenv("TEMPERATURE", "0.55")),
        top_p=float(os.getenv("TOP_P", "0.85")),
        max_tokens=int(os.getenv("MAX_TOKENS", "90")),
        user_cooldown_sec=int(os.getenv("USER_COOLDOWN_SEC", "15")),
        debug_traces=os.getenv("DEBUG_TRACES", "0").strip().lower() in {"1", "true", "yes", "on"},
        enable_debug_logs=os.getenv("ENABLE_DEBUG_LOGS", "1").strip().lower() in {"1", "true", "yes", "on"},
        enable_metrics=os.getenv("ENABLE_METRICS", "1").strip().lower() in {"1", "true", "yes", "on"},
        orchestration_log_path=os.getenv("ORCHESTRATION_LOG_PATH", "./orchestration.log").strip(),
        creator_user_id=_parse_optional_int(os.getenv("CREATOR_USER_ID", "")),
        creator_profile_path=os.getenv("CREATOR_PROFILE_PATH", "./creator_profile.json").strip(),
        join_verification_timeout_sec=int(os.getenv("JOIN_VERIFICATION_TIMEOUT_SEC", "90")),
        media_catalog_path=os.getenv("MEDIA_CATALOG_PATH", "./media_catalog.json").strip(),
        media_reply_prob=float(os.getenv("MEDIA_REPLY_PROB", "0.22")),
        media_reply_cooldown_sec=int(os.getenv("MEDIA_REPLY_COOLDOWN_SEC", "120")),
        knowledge_dir=os.getenv("KNOWLEDGE_DIR", "./knowledge").strip(),
        knowledge_max_files=int(os.getenv("KNOWLEDGE_MAX_FILES", "64")),
        knowledge_max_file_bytes=int(os.getenv("KNOWLEDGE_MAX_FILE_BYTES", "262144")),
    )
