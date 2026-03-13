import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


VOICE_MARKER_PART = (
    "(?:"
    "\u0433\u043e\u043b\u043e\u0441\u043e\u043c|"
    "\u0433\u043e\u043b\u043e\u0441\u043e\u0432\u044b\u043c\\s+\u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435\u043c|"
    "\u0433\u043e\u043b\u043e\u0441\u043e\u0432\u043e\u0435(?:\\s+\u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435)?|"
    "\u0432\u043e\u0439\u0441(?:\u043e\u043c)?|"
    "\u0433\u0441|"
    "\u043a\u0440\u0443\u0436\u043a\u043e\u043c"
    ")"
)

DIRECT_VOICE_PATTERNS = (
    re.compile(
        f"^(?:\u0437\u0430\u043f\u0438\u0448\u0438|\u0441\u0434\u0435\u043b\u0430\u0439|"
        f"\u0441\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u0443\u0439|\u043f\u0440\u0438\u0448\u043b\u0438|"
        f"\u043e\u0442\u043f\u0440\u0430\u0432\u044c)(?:\\s+\u043c\u043d\u0435)?\\s+{VOICE_MARKER_PART}"
        f"(?:\\s*(?::|-|,)\\s*|\\s+)(?P<content>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        f"^(?:\u043e\u0437\u0432\u0443\u0447\u044c|\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|"
        f"\u043f\u0440\u043e\u0438\u0437\u043d\u0435\u0441\u0438|\u0437\u0430\u0447\u0438\u0442\u0430\u0439|"
        f"\u043d\u0430\u0447\u0438\u0442\u0430\u0439)\\s+(?:\u043c\u043d\u0435\\s+)?(?P<content>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        f"^(?:\u0441\u043a\u0430\u0436\u0438|\u043f\u0440\u043e\u0438\u0437\u043d\u0435\u0441\u0438)\\s+"
        f"(?:\u043c\u043d\u0435\\s+)?{VOICE_MARKER_PART}(?:\\s*(?::|-|,)\\s*|\\s+)(?P<content>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        f"^{VOICE_MARKER_PART}(?:\\s*(?::|-|,)\\s*|\\s+)(?P<content>.+)$",
        re.IGNORECASE,
    ),
)

VOICE_REPLY_PREFIX_PATTERNS = (
    re.compile(
        f"^(?:\u043e\u0442\u0432\u0435\u0442\u044c|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438|"
        f"\u043e\u0431\u044a\u044f\u0441\u043d\u0438|\u043f\u0440\u043e\u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0438\u0440\u0443\u0439)"
        f"(?:\\s+\u043c\u043d\u0435)?\\s+{VOICE_MARKER_PART}"
        f"(?:\\s*(?::|-|,)\\s*|\\s*)(?P<content>.*)$",
        re.IGNORECASE,
    ),
    re.compile(
        f"^(?:\u0432|\u0432\u043e)\\s+{VOICE_MARKER_PART}"
        f"(?:\\s*(?::|-|,)\\s*|\\s*)(?P<content>.*)$",
        re.IGNORECASE,
    ),
)

VOICE_REPLY_SUFFIX_PATTERNS = (
    re.compile(
        f"^(?P<content>.+?)"
        f"(?:\\s*(?:,|:|-)\\s*|\\s+)"
        f"(?:(?:\u043e\u0442\u0432\u0435\u0442\u044c|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438|"
        f"\u043e\u0431\u044a\u044f\u0441\u043d\u0438|\u043f\u0440\u043e\u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0438\u0440\u0443\u0439)\\s+)?"
        f"{VOICE_MARKER_PART}$",
        re.IGNORECASE,
    ),
)

TELEGRAM_VOICE_EXTENSIONS = {"mp3", "ogg", "m4a"}


class ElevenLabsError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceRequest:
    requested: bool
    direct_text: bool
    content: str


@dataclass(frozen=True)
class VoiceSynthesisResult:
    audio_bytes: bytes
    filename: str


def _compact_spaces(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_voice_text(text: str) -> str:
    value = _compact_spaces(text).lower().replace("\u0451", "\u0435")
    value = re.sub(r"[\u00ab\u00bb\"'`]+", "", value)
    value = re.sub(r"\s*([,.:;!?-])\s*", r" \1 ", value)
    return _compact_spaces(value)


def _extract_voice_content(match: re.Match[str]) -> str:
    value = _compact_spaces(match.groupdict().get("content", ""))
    return re.sub(r"\s+([,.:;!?])", r"\1", value)


def _looks_like_voice_reply_request(text: str) -> bool:
    if not text:
        return False
    if not re.search(VOICE_MARKER_PART, text, re.IGNORECASE):
        return False
    if re.search(
        (
            "(?:^|\\s)"
            "(?:\u043e\u0442\u0432\u0435\u0442\u044c|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438|"
            "\u043e\u0431\u044a\u044f\u0441\u043d\u0438|\u043f\u0440\u043e\u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0438\u0440\u0443\u0439)"
            "(?:\\s|$)"
        ),
        text,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.fullmatch(
            f"(?:{VOICE_MARKER_PART}|"
            f"(?:\u0434\u0430\u0432\u0430\u0439|\u043c\u043e\u0436\u0435\u0448\u044c)\\s+{VOICE_MARKER_PART})",
            text,
            re.IGNORECASE,
        )
    )


def detect_voice_request(text: str) -> VoiceRequest | None:
    raw_value = _compact_spaces(text)
    if not raw_value:
        return None
    value = _normalize_voice_text(raw_value)

    for pattern in DIRECT_VOICE_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue
        content = _extract_voice_content(match)
        return VoiceRequest(requested=True, direct_text=True, content=content or raw_value)

    for pattern in VOICE_REPLY_PREFIX_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue
        return VoiceRequest(requested=True, direct_text=False, content=_extract_voice_content(match))

    for pattern in VOICE_REPLY_SUFFIX_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue
        return VoiceRequest(requested=True, direct_text=False, content=_extract_voice_content(match))

    if _looks_like_voice_reply_request(value):
        return VoiceRequest(requested=True, direct_text=False, content="")

    return None


def _output_extension(output_format: str) -> str:
    value = str(output_format or "").strip().lower()
    if not value:
        return "mp3"
    return value.split("_", 1)[0]


def _build_voice_settings(cfg: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if cfg.elevenlabs_stability is not None:
        settings["stability"] = float(cfg.elevenlabs_stability)
    if cfg.elevenlabs_similarity_boost is not None:
        settings["similarity_boost"] = float(cfg.elevenlabs_similarity_boost)
    if cfg.elevenlabs_style is not None:
        settings["style"] = float(cfg.elevenlabs_style)
    if cfg.elevenlabs_speed is not None:
        settings["speed"] = float(cfg.elevenlabs_speed)
    if cfg.elevenlabs_use_speaker_boost is not None:
        settings["use_speaker_boost"] = bool(cfg.elevenlabs_use_speaker_boost)
    return settings


class ElevenLabsTTS:
    def __init__(self, cfg: Any):
        self._api_key = cfg.elevenlabs_api_key
        self._base_url = cfg.elevenlabs_base_url.rstrip("/")
        self._model_id = cfg.elevenlabs_model_id
        self._voice_id = cfg.elevenlabs_voice_id
        self._output_format = cfg.elevenlabs_output_format
        self._language_code = cfg.elevenlabs_language_code
        self._enable_logging = cfg.elevenlabs_enable_logging
        self._optimize_streaming_latency = cfg.elevenlabs_optimize_streaming_latency
        self._apply_text_normalization = cfg.elevenlabs_apply_text_normalization
        self._seed = cfg.elevenlabs_seed
        self._timeout_sec = max(5, int(cfg.elevenlabs_timeout_sec))
        self._voice_settings = _build_voice_settings(cfg)
        self._extension = _output_extension(cfg.elevenlabs_output_format)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._voice_id)

    @property
    def extension(self) -> str:
        return self._extension

    def can_send_as_telegram_voice(self) -> bool:
        return self._extension in TELEGRAM_VOICE_EXTENSIONS

    def _build_url(self) -> str:
        query: dict[str, str] = {
            "output_format": self._output_format,
            "enable_logging": "true" if self._enable_logging else "false",
        }
        if self._optimize_streaming_latency is not None:
            query["optimize_streaming_latency"] = str(self._optimize_streaming_latency)
        query_string = parse.urlencode(query)
        voice_id = parse.quote(self._voice_id, safe="")
        return f"{self._base_url}/v1/text-to-speech/{voice_id}?{query_string}"

    def _build_payload(self, text: str) -> bytes:
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self._model_id,
        }
        if self._language_code:
            payload["language_code"] = self._language_code
        if self._apply_text_normalization:
            payload["apply_text_normalization"] = self._apply_text_normalization
        if self._seed is not None:
            payload["seed"] = int(self._seed)
        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings
        return json.dumps(payload).encode("utf-8")

    def _synthesize_sync(self, text: str) -> VoiceSynthesisResult:
        if not self.is_configured:
            raise ElevenLabsError("ElevenLabs is not configured")
        if not self.can_send_as_telegram_voice():
            raise ElevenLabsError(
                "ELEVENLABS_OUTPUT_FORMAT must produce mp3, ogg, or m4a for Telegram voice replies"
            )

        req = request.Request(
            self._build_url(),
            data=self._build_payload(text),
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/*",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_sec) as response:
                audio_bytes = response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ElevenLabsError(f"ElevenLabs HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise ElevenLabsError(f"ElevenLabs request failed: {exc.reason}") from exc

        if not audio_bytes:
            raise ElevenLabsError("ElevenLabs returned empty audio")

        return VoiceSynthesisResult(
            audio_bytes=audio_bytes,
            filename=f"alta-voice.{self._extension}",
        )

    async def synthesize(self, text: str) -> VoiceSynthesisResult:
        spoken_text = _compact_spaces(text)
        if not spoken_text:
            raise ElevenLabsError("Cannot synthesize empty text")
        return await asyncio.to_thread(self._synthesize_sync, spoken_text)
