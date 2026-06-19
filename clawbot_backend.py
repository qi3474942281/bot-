import asyncio
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from aiohttp import web


DB_FILE = "clawbot-data.sqlite3"
BACKEND_CONFIG_FILE = "backend_config.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

MEMORY_KINDS = {"fact", "stm", "ltm"}
MEMORY_SOURCES = {"user", "ai"}
DEFAULT_MEMORY_CONFIG = {
    "summaryModel": "",
    "factRounds": 30,
    "stmRounds": 30,
    "stmMinChars": 60,
    "stmMaxChars": 100,
    "ltmAfterStm": 6,
    "ltmMinChars": 300,
    "ltmMaxChars": 500,
    "factsAuto": True,
    "engineAuto": True,
}
DEFAULT_MEMORY_PROGRESS = {
    "factRounds": 0,
    "stmRounds": 0,
    "factCheckpoint": 0,
    "stmCheckpoint": 0,
    "pendingFact": False,
    "pendingStm": False,
    "pendingLtm": False,
    "clearStmBeforeNext": False,
    "lastError": "",
    "lastRoundMessageId": 0,
}
CHINA_TZ = timezone(timedelta(hours=8))
PROACTIVE_MODES = {
    "exact",
    "random_1_30",
    "random_30_60",
    "random_60_120",
    "random_120_180",
    "random_180_240",
    "random_240_300",
    "random_1_300",
}
DEFAULT_PROACTIVE_CONFIG = {
    "enabled": False,
    "mode": "random_60_120",
    "exactMinutes": 60,
    "windowStart": "08:00",
    "windowEnd": "23:00",
}
DEFAULT_PROACTIVE_PROGRESS = {
    "nextSendAt": "",
    "lastSentAt": "",
    "lastError": "",
}
DEFAULT_GENERAL_CONFIG = {
    "mergeWaitSeconds": 6,
    "currentModel": "",
    "splitReplyEnabled": True,
    "timeAwareEnabled": False,
    "weatherEnabled": False,
    "thinkingMode": "off",
}
THINKING_MODES = {"off", "web_only", "wechat_and_web"}


class VersionConflict(ValueError):
    pass


def _default_state() -> dict:
    character_id = "default"
    return {
        "activeCharacterId": character_id,
        "characters": [
            {
                "id": character_id,
                "name": "默认 ClawBot",
                "persona": {"description": "", "traits": "", "examples": ""},
                "memories": [],
                "promptMemories": [],
            }
        ],
    }


def _clean_text(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _int_setting(value, default, minimum, maximum):
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(result, maximum))


def _time_setting(value, default):
    text = str(value or "")
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError:
        return default
    return parsed.strftime("%H:%M")


def _normalize_proactive_config(value) -> dict:
    raw = value if isinstance(value, dict) else {}
    mode = _clean_text(raw.get("mode"), 30)
    if mode not in PROACTIVE_MODES:
        mode = DEFAULT_PROACTIVE_CONFIG["mode"]
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": mode,
        "exactMinutes": _int_setting(raw.get("exactMinutes"), 60, 1, 1440),
        "windowStart": _time_setting(raw.get("windowStart"), "08:00"),
        "windowEnd": _time_setting(raw.get("windowEnd"), "23:00"),
    }


def _normalize_proactive_progress(value) -> dict:
    raw = value if isinstance(value, dict) else {}
    return {
        "nextSendAt": _clean_text(raw.get("nextSendAt"), 80),
        "lastSentAt": _clean_text(raw.get("lastSentAt"), 80),
        "lastError": _clean_text(raw.get("lastError"), 1000),
    }


def _normalize_general_config(value) -> dict:
    raw = value if isinstance(value, dict) else {}
    thinking_mode = _clean_text(raw.get("thinkingMode"), 40)
    if thinking_mode not in THINKING_MODES:
        thinking_mode = DEFAULT_GENERAL_CONFIG["thinkingMode"]
    return {
        "mergeWaitSeconds": _int_setting(
            raw.get("mergeWaitSeconds"), 6, 1, 30
        ),
        "currentModel": _clean_text(raw.get("currentModel"), 100),
        "splitReplyEnabled": bool(raw.get("splitReplyEnabled", True)),
        "timeAwareEnabled": bool(raw.get("timeAwareEnabled", False)),
        "weatherEnabled": bool(raw.get("weatherEnabled", False)),
        "thinkingMode": thinking_mode,
    }


def _normalize_affection_value(value) -> int:
    if isinstance(value, dict):
        value = value.get("value")
    return _int_setting(value, 0, 0, 500)


def _parse_utc(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _window_bounds(now_utc, config):
    local_now = now_utc.astimezone(CHINA_TZ)
    start_time = datetime.strptime(config["windowStart"], "%H:%M").time()
    end_time = datetime.strptime(config["windowEnd"], "%H:%M").time()
    start = datetime.combine(local_now.date(), start_time, CHINA_TZ)
    end = datetime.combine(local_now.date(), end_time, CHINA_TZ)
    if start_time == end_time:
        return local_now, local_now + timedelta(days=1)
    if start_time < end_time:
        if local_now < start:
            return start, end
        if local_now >= end:
            return start + timedelta(days=1), end + timedelta(days=1)
        return start, end
    if local_now >= start:
        return start, end + timedelta(days=1)
    if local_now < end:
        return start - timedelta(days=1), end
    return start, end + timedelta(days=1)


def proactive_allowed_at(moment_utc, config):
    local = moment_utc.astimezone(CHINA_TZ)
    start = datetime.strptime(config["windowStart"], "%H:%M").time()
    end = datetime.strptime(config["windowEnd"], "%H:%M").time()
    current = local.time().replace(second=0, microsecond=0)
    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


def defer_to_proactive_window(moment_utc, config):
    if proactive_allowed_at(moment_utc, config):
        return moment_utc
    start, _ = _window_bounds(moment_utc, config)
    return start.astimezone(timezone.utc)


def _normalize_memory_config(value) -> dict:
    raw = value if isinstance(value, dict) else {}
    result = {
        "summaryModel": _clean_text(raw.get("summaryModel"), 100),
        "factRounds": _int_setting(raw.get("factRounds"), 30, 1, 10000),
        "stmRounds": _int_setting(raw.get("stmRounds"), 30, 1, 10000),
        "stmMinChars": _int_setting(raw.get("stmMinChars"), 60, 10, 5000),
        "stmMaxChars": _int_setting(raw.get("stmMaxChars"), 100, 10, 10000),
        "ltmAfterStm": _int_setting(raw.get("ltmAfterStm"), 6, 1, 1000),
        "ltmMinChars": _int_setting(raw.get("ltmMinChars"), 300, 20, 20000),
        "ltmMaxChars": _int_setting(raw.get("ltmMaxChars"), 500, 20, 30000),
        "factsAuto": bool(raw.get("factsAuto", True)),
        "engineAuto": bool(raw.get("engineAuto", True)),
    }
    if result["stmMinChars"] > result["stmMaxChars"]:
        raise ValueError("STM 最小字数不能大于最大字数")
    if result["ltmMinChars"] > result["ltmMaxChars"]:
        raise ValueError("LTM 最小字数不能大于最大字数")
    return result


def _normalize_state(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("state must be an object")
    raw_characters = value.get("characters")
    if not isinstance(raw_characters, list) or not raw_characters:
        raise ValueError("at least one character is required")

    characters = []
    seen_ids = set()
    for raw in raw_characters[:50]:
        if not isinstance(raw, dict):
            continue
        character_id = _clean_text(raw.get("id"), 100) or secrets.token_hex(8)
        if character_id in seen_ids:
            continue
        seen_ids.add(character_id)
        persona = raw.get("persona") if isinstance(raw.get("persona"), dict) else {}
        memories = raw.get("memories") if isinstance(raw.get("memories"), list) else []
        prompt_memories = (
            raw.get("promptMemories")
            if isinstance(raw.get("promptMemories"), list)
            else []
        )
        characters.append(
            {
                "id": character_id,
                "name": _clean_text(raw.get("name"), 100) or "未命名 ClawBot",
                "persona": {
                    "description": _clean_text(persona.get("description"), 12000),
                    "traits": _clean_text(persona.get("traits"), 4000),
                    "examples": _clean_text(persona.get("examples"), 8000),
                },
                "memories": [
                    _clean_text(item, 2000)
                    for item in memories[:200]
                    if _clean_text(item, 2000)
                ],
                "promptMemories": [
                    _clean_text(item, 2000)
                    for item in prompt_memories[:200]
                    if _clean_text(item, 2000)
                ],
            }
        )
    if not characters:
        raise ValueError("at least one valid character is required")
    active_id = _clean_text(value.get("activeCharacterId"), 100)
    if active_id not in {item["id"] for item in characters}:
        active_id = characters[0]["id"]
    return {"activeCharacterId": active_id, "characters": characters}


class ClawBotStore:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.lock = threading.RLock()
        self._memory_connection = None
        if db_file == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self):
        if self._memory_connection is not None:
            return self._memory_connection
        connection = sqlite3.connect(self.db_file, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self.lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    thinking_summary TEXT NOT NULL DEFAULT '',
                    count_for_memory INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_profiles (
                    character_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    migrated INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL DEFAULT 0,
                    time_label TEXT NOT NULL DEFAULT '',
                    scene TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_entries_character_kind
                ON memory_entries(character_id, kind, sequence_no, id);
                CREATE TABLE IF NOT EXISTS proactive_profiles (
                    character_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS general_profiles (
                    character_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS affection_profiles (
                    character_id TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS affection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    delta INTEGER NOT NULL,
                    value_after INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS affection_history_character
                ON affection_history(character_id, id DESC);
                """
            )
            message_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "count_for_memory" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN count_for_memory INTEGER NOT NULL DEFAULT 1"
                )
            if "thinking_summary" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN thinking_summary TEXT NOT NULL DEFAULT ''"
                )
            existing = connection.execute(
                "SELECT 1 FROM settings WHERE key = 'state'"
            ).fetchone()
            if not existing:
                connection.execute(
                    "INSERT INTO settings(key, value) VALUES('state', ?)",
                    (json.dumps(_default_state(), ensure_ascii=False),),
                )
            self._ensure_profiles(connection, self._state_from_connection(connection))

    def _state_from_connection(self, connection) -> dict:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = 'state'"
        ).fetchone()
        return _normalize_state(json.loads(row["value"]))

    def _ensure_profiles(self, connection, state):
        now = datetime.now(timezone.utc).isoformat()
        for character in state["characters"]:
            row = connection.execute(
                "SELECT migrated FROM memory_profiles WHERE character_id = ?",
                (character["id"],),
            ).fetchone()
            if not row:
                latest_message_id = self._latest_message_id(
                    connection, character["id"]
                )
                progress = dict(DEFAULT_MEMORY_PROGRESS)
                progress.update(
                    {
                        "factCheckpoint": latest_message_id,
                        "stmCheckpoint": latest_message_id,
                        "lastRoundMessageId": latest_message_id,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO memory_profiles(
                        character_id, config_json, progress_json, version, migrated, updated_at
                    ) VALUES(?, ?, ?, 1, 0, ?)
                    """,
                    (
                        character["id"],
                        json.dumps(DEFAULT_MEMORY_CONFIG, ensure_ascii=False),
                        json.dumps(progress, ensure_ascii=False),
                        now,
                    ),
                )
                row = {"migrated": 0}
            if not row["migrated"]:
                for content in character.get("memories", []):
                    duplicate = connection.execute(
                        """
                        SELECT 1 FROM memory_entries
                        WHERE character_id = ? AND kind = 'fact' AND content = ?
                        """,
                        (character["id"], content),
                    ).fetchone()
                    if not duplicate:
                        connection.execute(
                            """
                            INSERT INTO memory_entries(
                                character_id, kind, content, source, created_at, updated_at
                            ) VALUES(?, 'fact', ?, 'user', ?, ?)
                            """,
                            (character["id"], content, now, now),
                        )
                connection.execute(
                    "UPDATE memory_profiles SET migrated = 1 WHERE character_id = ?",
                    (character["id"],),
                )
            proactive = connection.execute(
                "SELECT 1 FROM proactive_profiles WHERE character_id = ?",
                (character["id"],),
            ).fetchone()
            if not proactive:
                connection.execute(
                    """
                    INSERT INTO proactive_profiles(
                        character_id, config_json, progress_json, version, updated_at
                    ) VALUES(?, ?, ?, 1, ?)
                    """,
                    (
                        character["id"],
                        json.dumps(DEFAULT_PROACTIVE_CONFIG, ensure_ascii=False),
                        json.dumps(DEFAULT_PROACTIVE_PROGRESS, ensure_ascii=False),
                        now,
                    ),
                )
            general = connection.execute(
                "SELECT 1 FROM general_profiles WHERE character_id = ?",
                (character["id"],),
            ).fetchone()
            if not general:
                connection.execute(
                    """
                    INSERT INTO general_profiles(
                        character_id, config_json, version, updated_at
                    ) VALUES(?, ?, 1, ?)
                    """,
                    (
                        character["id"],
                        json.dumps(DEFAULT_GENERAL_CONFIG, ensure_ascii=False),
                        now,
                    ),
                )
            affection = connection.execute(
                "SELECT 1 FROM affection_profiles WHERE character_id = ?",
                (character["id"],),
            ).fetchone()
            if not affection:
                connection.execute(
                    """
                    INSERT INTO affection_profiles(
                        character_id, value, version, updated_at
                    ) VALUES(?, 0, 1, ?)
                    """,
                    (character["id"], now),
                )

    def get_state(self) -> dict:
        with self.lock, self._connect() as connection:
            state = self._state_from_connection(connection)
            self._ensure_profiles(connection, state)
        return state

    def save_state(self, value) -> dict:
        state = _normalize_state(value)
        payload = json.dumps(state, ensure_ascii=False)
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value) VALUES('state', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (payload,),
            )
            self._ensure_profiles(connection, state)
        return state

    def active_character(self) -> dict:
        state = self.get_state()
        return next(
            item
            for item in state["characters"]
            if item["id"] == state["activeCharacterId"]
        )

    def character(self, character_id: str) -> dict:
        state = self.get_state()
        for item in state["characters"]:
            if item["id"] == character_id:
                return item
        raise ValueError("character not found")

    def general_snapshot(self, character_id: str) -> dict:
        with self.lock, self._connect() as connection:
            self._ensure_profiles(connection, self._state_from_connection(connection))
            row = connection.execute(
                "SELECT * FROM general_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character general profile not found")
            config = dict(DEFAULT_GENERAL_CONFIG)
            config.update(json.loads(row["config_json"]))
        return {
            "characterId": character_id,
            "config": _normalize_general_config(config),
            "version": row["version"],
        }

    def save_general_config(
        self, character_id, value, expected_version, available_models=None
    ) -> dict:
        config = _normalize_general_config(value)
        models = available_models or {}
        if config["currentModel"] and config["currentModel"] not in models:
            raise ValueError("chat model must come from the current model list")
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM general_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character general profile not found")
            if expected_version is None or int(expected_version) != row["version"]:
                raise VersionConflict("通用设置已更新，请刷新后重试")
            connection.execute(
                """
                UPDATE general_profiles
                SET config_json = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(config, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )
        return self.general_snapshot(character_id)

    def affection_snapshot(self, character_id: str) -> dict:
        with self.lock, self._connect() as connection:
            self._ensure_profiles(connection, self._state_from_connection(connection))
            row = connection.execute(
                "SELECT * FROM affection_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character affection profile not found")
            history = connection.execute(
                """
                SELECT id, delta, value_after, reason, created_at
                FROM affection_history
                WHERE character_id = ?
                ORDER BY id DESC LIMIT 10
                """,
                (character_id,),
            ).fetchall()
        return {
            "characterId": character_id,
            "value": _normalize_affection_value(row["value"]),
            "history": [dict(item) for item in history],
            "version": row["version"],
        }

    def save_affection(self, character_id, value, expected_version) -> dict:
        normalized = _normalize_affection_value(value)
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM affection_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character affection profile not found")
            if expected_version is None or int(expected_version) != row["version"]:
                raise VersionConflict("好感度已更新，请刷新后重试")
            connection.execute(
                """
                UPDATE affection_profiles
                SET value = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    normalized,
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )
        return self.affection_snapshot(character_id)

    def apply_affection_delta(self, character_id, delta, reason) -> dict:
        delta = _int_setting(delta, 0, -500, 500)
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM affection_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character affection profile not found")
            old_value = _normalize_affection_value(row["value"])
            new_value = max(0, min(500, old_value + delta))
            applied_delta = new_value - old_value
            connection.execute(
                """
                UPDATE affection_profiles
                SET value = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (new_value, now, character_id),
            )
            if applied_delta:
                connection.execute(
                    """
                    INSERT INTO affection_history(
                        character_id, delta, value_after, reason, created_at
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        character_id,
                        applied_delta,
                        new_value,
                        _clean_text(reason, 1000),
                        now,
                    ),
                )
        return self.affection_snapshot(character_id)

    def proactive_snapshot(self, character_id: str) -> dict:
        with self.lock, self._connect() as connection:
            self._ensure_profiles(connection, self._state_from_connection(connection))
            row = connection.execute(
                "SELECT * FROM proactive_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character proactive profile not found")
            config = dict(DEFAULT_PROACTIVE_CONFIG)
            config.update(json.loads(row["config_json"]))
            progress = dict(DEFAULT_PROACTIVE_PROGRESS)
            progress.update(json.loads(row["progress_json"]))
        return {
            "characterId": character_id,
            "config": _normalize_proactive_config(config),
            "progress": _normalize_proactive_progress(progress),
            "version": row["version"],
        }

    def save_proactive_config(self, character_id, value, expected_version) -> dict:
        config = _normalize_proactive_config(value)
        now = datetime.now(timezone.utc)
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM proactive_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character proactive profile not found")
            if expected_version is None or int(expected_version) != row["version"]:
                raise VersionConflict("主动消息设置已更新，请刷新后重试")
            old_config = _normalize_proactive_config(json.loads(row["config_json"]))
            progress = _normalize_proactive_progress(json.loads(row["progress_json"]))
            if old_config["enabled"] and not config["enabled"]:
                progress["nextSendAt"] = ""
                progress["lastError"] = ""
            elif not old_config["enabled"] and config["enabled"]:
                progress["nextSendAt"] = ""
                progress["lastError"] = ""
            connection.execute(
                """
                UPDATE proactive_profiles
                SET config_json = ?, progress_json = ?, version = version + 1,
                    updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(progress, ensure_ascii=False),
                    now.isoformat(),
                    character_id,
                ),
            )
        return self.proactive_snapshot(character_id)

    def set_proactive_schedule(self, character_id, next_send_at) -> dict:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT progress_json FROM proactive_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character proactive profile not found")
            progress = _normalize_proactive_progress(json.loads(row["progress_json"]))
            progress["nextSendAt"] = (
                next_send_at.astimezone(timezone.utc).isoformat()
                if next_send_at
                else ""
            )
            connection.execute(
                """
                UPDATE proactive_profiles SET progress_json = ?, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )
        return self.proactive_snapshot(character_id)

    def mark_proactive_sent(self, character_id, next_send_at, sent_at=None):
        sent_at = sent_at or datetime.now(timezone.utc)
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT progress_json FROM proactive_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            progress = _normalize_proactive_progress(json.loads(row["progress_json"]))
            progress.update(
                {
                    "nextSendAt": next_send_at.astimezone(timezone.utc).isoformat(),
                    "lastSentAt": sent_at.astimezone(timezone.utc).isoformat(),
                    "lastError": "",
                }
            )
            connection.execute(
                "UPDATE proactive_profiles SET progress_json = ?, updated_at = ? WHERE character_id = ?",
                (
                    json.dumps(progress, ensure_ascii=False),
                    sent_at.astimezone(timezone.utc).isoformat(),
                    character_id,
                ),
            )

    def mark_proactive_error(self, character_id, message, retry_at):
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT progress_json FROM proactive_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            progress = _normalize_proactive_progress(json.loads(row["progress_json"]))
            progress["lastError"] = _clean_text(message, 1000)
            progress["nextSendAt"] = retry_at.astimezone(timezone.utc).isoformat()
            connection.execute(
                "UPDATE proactive_profiles SET progress_json = ?, updated_at = ? WHERE character_id = ?",
                (
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )

    def proactive_due(self, character_id, now=None):
        now = now or datetime.now(timezone.utc)
        snapshot = self.proactive_snapshot(character_id)
        if not snapshot["config"]["enabled"]:
            return False
        next_send = _parse_utc(snapshot["progress"]["nextSendAt"])
        return bool(next_send and next_send <= now)

    def _profile(self, connection, character_id):
        row = connection.execute(
            "SELECT * FROM memory_profiles WHERE character_id = ?",
            (character_id,),
        ).fetchone()
        if not row:
            raise ValueError("character memory profile not found")
        config = dict(DEFAULT_MEMORY_CONFIG)
        config.update(json.loads(row["config_json"]))
        progress = dict(DEFAULT_MEMORY_PROGRESS)
        progress.update(json.loads(row["progress_json"]))
        return row, _normalize_memory_config(config), progress

    def memory_snapshot(self, character_id: str) -> dict:
        with self.lock, self._connect() as connection:
            self._ensure_profiles(connection, self._state_from_connection(connection))
            row, config, progress = self._profile(connection, character_id)
            entries = connection.execute(
                """
                SELECT id, kind, sequence_no, time_label, scene, content, source,
                       created_at, updated_at
                FROM memory_entries
                WHERE character_id = ?
                ORDER BY CASE kind WHEN 'fact' THEN 0 WHEN 'stm' THEN 1 ELSE 2 END,
                         sequence_no, id
                """,
                (character_id,),
            ).fetchall()
        grouped = {"facts": [], "stm": [], "ltm": []}
        for item in map(dict, entries):
            grouped[{"fact": "facts", "stm": "stm", "ltm": "ltm"}[item["kind"]]].append(item)
        return {
            "characterId": character_id,
            "config": config,
            "progress": progress,
            "version": row["version"],
            **grouped,
        }

    def save_memory_config(self, character_id, config, expected_version) -> dict:
        normalized = _normalize_memory_config(config)
        with self.lock, self._connect() as connection:
            row, old_config, progress = self._profile(connection, character_id)
            if int(expected_version) != row["version"]:
                raise VersionConflict("记忆数据已更新，请刷新后重试")
            if old_config["factsAuto"] != normalized["factsAuto"]:
                progress["factRounds"] = 0
                progress["pendingFact"] = False
                progress["factCheckpoint"] = self._latest_message_id(connection, character_id)
            if old_config["engineAuto"] != normalized["engineAuto"]:
                progress["stmRounds"] = 0
                progress["pendingStm"] = False
                progress["pendingLtm"] = False
                progress["stmCheckpoint"] = self._latest_message_id(connection, character_id)
            connection.execute(
                """
                UPDATE memory_profiles
                SET config_json = ?, progress_json = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(normalized, ensure_ascii=False),
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )
        return self.memory_snapshot(character_id)

    def _assert_version(self, connection, character_id, expected_version):
        if expected_version is None:
            return
        row = connection.execute(
            "SELECT version FROM memory_profiles WHERE character_id = ?",
            (character_id,),
        ).fetchone()
        if not row or int(expected_version) != row["version"]:
            raise VersionConflict("记忆数据已更新，请刷新后重试")

    def add_memory_entry(self, character_id, value, expected_version=None) -> dict:
        kind = _clean_text(value.get("kind"), 20)
        if kind not in MEMORY_KINDS:
            raise ValueError("invalid memory kind")
        content = _clean_text(value.get("content"), 30000)
        if not content:
            raise ValueError("memory content is required")
        source = value.get("source") if value.get("source") in MEMORY_SOURCES else "user"
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as connection:
            self._assert_version(connection, character_id, expected_version)
            sequence = 0
            if kind in {"stm", "ltm"}:
                sequence = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1
                    FROM memory_entries WHERE character_id = ? AND kind = ?
                    """,
                    (character_id, kind),
                ).fetchone()[0]
            cursor = connection.execute(
                """
                INSERT INTO memory_entries(
                    character_id, kind, sequence_no, time_label, scene, content,
                    source, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_id,
                    kind,
                    sequence,
                    _clean_text(value.get("timeLabel"), 500),
                    _clean_text(value.get("scene"), 1000),
                    content,
                    source,
                    now,
                    now,
                ),
            )
            self._bump_version(connection, character_id)
            entry_id = cursor.lastrowid
        return self.get_memory_entry(character_id, entry_id)

    def get_memory_entry(self, character_id, entry_id):
        with self.lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, sequence_no, time_label, scene, content, source,
                       created_at, updated_at
                FROM memory_entries WHERE character_id = ? AND id = ?
                """,
                (character_id, entry_id),
            ).fetchone()
        if not row:
            raise ValueError("memory entry not found")
        return dict(row)

    def update_memory_entry(
        self, character_id, entry_id, value, expected_version=None
    ) -> dict:
        content = _clean_text(value.get("content"), 30000)
        if not content:
            raise ValueError("memory content is required")
        with self.lock, self._connect() as connection:
            self._assert_version(connection, character_id, expected_version)
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET time_label = ?, scene = ?, content = ?, source = 'user', updated_at = ?
                WHERE character_id = ? AND id = ?
                """,
                (
                    _clean_text(value.get("timeLabel"), 500),
                    _clean_text(value.get("scene"), 1000),
                    content,
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                    int(entry_id),
                ),
            )
            if not cursor.rowcount:
                raise ValueError("memory entry not found")
            self._bump_version(connection, character_id)
        return self.get_memory_entry(character_id, entry_id)

    def delete_memory_entry(
        self, character_id, entry_id, expected_version=None
    ):
        with self.lock, self._connect() as connection:
            self._assert_version(connection, character_id, expected_version)
            cursor = connection.execute(
                "DELETE FROM memory_entries WHERE character_id = ? AND id = ?",
                (character_id, int(entry_id)),
            )
            if not cursor.rowcount:
                raise ValueError("memory entry not found")
            self._renumber_entries(connection, character_id)
            self._bump_version(connection, character_id)

    def clear_memory(self, character_id, expected_version=None) -> dict:
        with self.lock, self._connect() as connection:
            self._assert_version(connection, character_id, expected_version)
            row, config, _progress = self._profile(connection, character_id)
            progress = dict(DEFAULT_MEMORY_PROGRESS)
            latest_id = self._latest_message_id(connection, character_id)
            progress["factCheckpoint"] = latest_id
            progress["stmCheckpoint"] = latest_id
            progress["lastRoundMessageId"] = latest_id
            connection.execute(
                "DELETE FROM memory_entries WHERE character_id = ?",
                (character_id,),
            )
            connection.execute(
                """
                UPDATE memory_profiles
                SET progress_json = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )
        return self.memory_snapshot(character_id)

    def delete_message(self, message_id):
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM messages WHERE id = ?",
                (int(message_id),),
            )
            if not cursor.rowcount:
                raise ValueError("message not found")

    def clear_messages(self, character_id):
        with self.lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE character_id = ?",
                (character_id,),
            )
        return self.history_for_web()

    def reset_affection(self, character_id, expected_version=None) -> dict:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM affection_profiles WHERE character_id = ?",
                (character_id,),
            ).fetchone()
            if not row:
                raise ValueError("character affection profile not found")
            if expected_version is not None and int(expected_version) != row["version"]:
                raise VersionConflict("好感度已更新，请刷新后重试")
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE affection_profiles
                SET value = 0, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (now, character_id),
            )
            connection.execute(
                "DELETE FROM affection_history WHERE character_id = ?",
                (character_id,),
            )
        return self.affection_snapshot(character_id)

    def _renumber_entries(self, connection, character_id):
        for kind in ("stm", "ltm"):
            rows = connection.execute(
                """
                SELECT id FROM memory_entries
                WHERE character_id = ? AND kind = ? ORDER BY sequence_no, id
                """,
                (character_id, kind),
            ).fetchall()
            for number, row in enumerate(rows, 1):
                connection.execute(
                    "UPDATE memory_entries SET sequence_no = ? WHERE id = ?",
                    (number, row["id"]),
                )

    def _bump_version(self, connection, character_id):
        connection.execute(
            """
            UPDATE memory_profiles
            SET version = version + 1, updated_at = ?
            WHERE character_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), character_id),
        )

    def build_prompt(self, fallback_prompt: str, character_id: str | None = None) -> str:
        character = (
            self.character(character_id) if character_id else self.active_character()
        )
        persona = character["persona"]
        snapshot = self.memory_snapshot(character["id"])
        affection = self.affection_snapshot(character["id"])
        sections = [fallback_prompt, f"你当前扮演：{character['name']}。"]
        sections.append(
            "【当前好感度】\n"
            f"{affection['value']} / 500。请让亲近程度自然反映这个数值，"
            "不要在回复中直接提到好感度数值或系统设置。"
        )
        if persona["description"]:
            sections.append(f"【核心人设】\n{persona['description']}")
        if persona["traits"]:
            sections.append(f"【性格特点】\n{persona['traits']}")
        if persona["examples"]:
            sections.append(f"【示例对话】\n{persona['examples']}")
        if character["promptMemories"]:
            sections.append(
                "【必须遵守的规则】\n"
                + "\n".join(f"- {item}" for item in character["promptMemories"])
            )
        if snapshot["facts"]:
            sections.append(
                "【需要记住的关键事实】\n"
                + "\n".join(f"- {item['content']}" for item in snapshot["facts"])
            )
        if snapshot["stm"]:
            sections.append(
                "【短期记忆】\n"
                + "\n".join(
                    f"{item['sequence_no']}. {item['time_label']} | {item['scene']} | {item['content']}"
                    for item in snapshot["stm"]
                )
            )
        if snapshot["ltm"]:
            sections.append(
                "【长期记忆】\n"
                + "\n".join(
                    f"{item['sequence_no']}. {item['time_label']} | {item['scene']} | {item['content']}"
                    for item in snapshot["ltm"]
                )
            )
        return "\n\n".join(section for section in sections if section)

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        count_for_memory: bool = True,
        character_id: str | None = None,
        thinking_summary: str = "",
    ):
        character = (
            self.character(character_id) if character_id else self.active_character()
        )
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    character_id, user_id, role, content, thinking_summary,
                    count_for_memory, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character["id"],
                    _clean_text(user_id, 200),
                    role,
                    _clean_text(content, 20000),
                    _clean_text(thinking_summary, 2000),
                    1 if count_for_memory else 0,
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM messages
                WHERE id NOT IN (
                    SELECT id FROM messages ORDER BY id DESC LIMIT 5000
                )
                """
            )
        return cursor.lastrowid

    def recent_history(
        self,
        user_id: str,
        limit: int = 20,
        character_id: str | None = None,
        include_timestamps: bool = False,
    ) -> list[dict]:
        character = (
            self.character(character_id) if character_id else self.active_character()
        )
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at FROM messages
                WHERE character_id = ? AND user_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (character["id"], user_id, max(1, min(limit, 100))),
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = {"role": row["role"], "content": row["content"]}
            if include_timestamps:
                item["created_at"] = row["created_at"]
            result.append(item)
        return result

    def history_for_web(self, limit: int = 300) -> list[dict]:
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, character_id, user_id, role, content,
                       thinking_summary, created_at
                FROM messages ORDER BY id DESC LIMIT ?
                """,
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def _latest_message_id(self, connection, character_id):
        return connection.execute(
            """
            SELECT COALESCE(MAX(id), 0) FROM messages
            WHERE character_id = ? AND count_for_memory = 1
            """,
            (character_id,),
        ).fetchone()[0]

    def _task_messages(self, connection, character_id, user_id, checkpoint):
        rows = connection.execute(
            """
            SELECT id, role, content, created_at FROM messages
            WHERE character_id = ? AND user_id = ? AND id > ?
            ORDER BY id
            """,
            (character_id, user_id, checkpoint),
        ).fetchall()
        return [dict(row) for row in rows]

    def prepare_memory_tasks(self, character_id, user_id) -> list[dict]:
        tasks = []
        with self.lock, self._connect() as connection:
            row, config, progress = self._profile(connection, character_id)
            changed = False
            latest_id = self._latest_message_id(connection, character_id)
            if latest_id > progress["lastRoundMessageId"]:
                progress["lastRoundMessageId"] = latest_id
                changed = True
                if config["factsAuto"] and not progress["pendingFact"]:
                    progress["factRounds"] += 1
                    if progress["factRounds"] >= config["factRounds"]:
                        progress["pendingFact"] = True
                if config["engineAuto"] and not progress["pendingStm"]:
                    progress["stmRounds"] += 1
                    if (
                        progress["stmRounds"] >= config["stmRounds"]
                        and not progress["pendingLtm"]
                    ):
                        progress["pendingStm"] = True
            if progress["pendingFact"]:
                ai_facts = connection.execute(
                    """
                    SELECT id, content FROM memory_entries
                    WHERE character_id = ? AND kind = 'fact' AND source = 'ai'
                    ORDER BY id
                    """,
                    (character_id,),
                ).fetchall()
                tasks.append(
                    {
                        "kind": "fact",
                        "characterId": character_id,
                        "messages": self._task_messages(
                            connection, character_id, user_id, progress["factCheckpoint"]
                        ),
                        "existingAiFacts": [dict(item) for item in ai_facts],
                        "config": config,
                        "latestMessageId": latest_id,
                    }
                )
            if progress["pendingStm"]:
                if progress["clearStmBeforeNext"]:
                    connection.execute(
                        "DELETE FROM memory_entries WHERE character_id = ? AND kind = 'stm'",
                        (character_id,),
                    )
                    progress["clearStmBeforeNext"] = False
                    changed = True
                tasks.append(
                    {
                        "kind": "stm",
                        "characterId": character_id,
                        "messages": self._task_messages(
                            connection, character_id, user_id, progress["stmCheckpoint"]
                        ),
                        "config": config,
                        "latestMessageId": latest_id,
                    }
                )
            if progress["pendingLtm"]:
                stm_rows = connection.execute(
                    """
                    SELECT id, kind, sequence_no, time_label, scene, content, source,
                           created_at, updated_at
                    FROM memory_entries
                    WHERE character_id = ? AND kind = 'stm'
                    ORDER BY sequence_no, id
                    """,
                    (character_id,),
                ).fetchall()
                tasks.append(
                    {
                        "kind": "ltm",
                        "characterId": character_id,
                        "stm": [dict(item) for item in stm_rows],
                        "config": config,
                    }
                )
            if changed:
                connection.execute(
                    """
                    UPDATE memory_profiles
                    SET progress_json = ?, version = version + 1, updated_at = ?
                    WHERE character_id = ?
                    """,
                    (
                        json.dumps(progress, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                        character_id,
                    ),
                )
        return tasks

    def complete_fact_task(self, task, result):
        action = result.get("action")
        content = _clean_text(result.get("content"), 30000)
        if action not in {"add", "merge"} or not content:
            raise ValueError("invalid fact summary result")
        with self.lock, self._connect() as connection:
            if action == "merge":
                entry_id = int(result.get("entryId") or 0)
                row = connection.execute(
                    """
                    SELECT 1 FROM memory_entries
                    WHERE id = ? AND character_id = ? AND kind = 'fact' AND source = 'ai'
                    """,
                    (entry_id, task["characterId"]),
                ).fetchone()
                if not row:
                    raise ValueError("AI fact to merge was not found")
                connection.execute(
                    """
                    UPDATE memory_entries SET content = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (content, datetime.now(timezone.utc).isoformat(), entry_id),
                )
            else:
                self._insert_generated_entry(connection, task["characterId"], "fact", result)
            self._complete_task_progress(
                connection, task["characterId"], "fact", task["latestMessageId"]
            )

    def complete_stm_task(self, task, result) -> bool:
        content = _clean_text(result.get("content"), 30000)
        length = len(content)
        if not content or not (
            task["config"]["stmMinChars"] <= length <= task["config"]["stmMaxChars"]
        ):
            raise ValueError("STM summary length is outside configured range")
        should_create_ltm = False
        with self.lock, self._connect() as connection:
            self._insert_generated_entry(connection, task["characterId"], "stm", result)
            self._complete_task_progress(
                connection, task["characterId"], "stm", task["latestMessageId"]
            )
            count = connection.execute(
                """
                SELECT COUNT(*) FROM memory_entries
                WHERE character_id = ? AND kind = 'stm'
                """,
                (task["characterId"],),
            ).fetchone()[0]
            should_create_ltm = count >= task["config"]["ltmAfterStm"]
            if should_create_ltm:
                row, config, progress = self._profile(
                    connection, task["characterId"]
                )
                progress["pendingLtm"] = True
                connection.execute(
                    """
                    UPDATE memory_profiles
                    SET progress_json = ?, version = version + 1, updated_at = ?
                    WHERE character_id = ?
                    """,
                    (
                        json.dumps(progress, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                        task["characterId"],
                    ),
                )
        return should_create_ltm

    def prepare_ltm_task(self, character_id) -> dict:
        snapshot = self.memory_snapshot(character_id)
        return {
            "kind": "ltm",
            "characterId": character_id,
            "stm": snapshot["stm"],
            "config": snapshot["config"],
        }

    def complete_ltm_task(self, task, result):
        content = _clean_text(result.get("content"), 30000)
        length = len(content)
        if not content or not (
            task["config"]["ltmMinChars"] <= length <= task["config"]["ltmMaxChars"]
        ):
            raise ValueError("LTM summary length is outside configured range")
        with self.lock, self._connect() as connection:
            self._insert_generated_entry(connection, task["characterId"], "ltm", result)
            row, config, progress = self._profile(connection, task["characterId"])
            progress["pendingLtm"] = False
            progress["clearStmBeforeNext"] = True
            progress["lastError"] = ""
            connection.execute(
                """
                UPDATE memory_profiles
                SET progress_json = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    task["characterId"],
                ),
            )

    def _insert_generated_entry(self, connection, character_id, kind, result):
        sequence = 0
        if kind in {"stm", "ltm"}:
            sequence = connection.execute(
                """
                SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM memory_entries
                WHERE character_id = ? AND kind = ?
                """,
                (character_id, kind),
            ).fetchone()[0]
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO memory_entries(
                character_id, kind, sequence_no, time_label, scene, content,
                source, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, 'ai', ?, ?)
            """,
            (
                character_id,
                kind,
                sequence,
                _clean_text(result.get("timeLabel"), 500),
                _clean_text(result.get("scene"), 1000),
                _clean_text(result.get("content"), 30000),
                now,
                now,
            ),
        )

    def _complete_task_progress(self, connection, character_id, kind, checkpoint):
        row, config, progress = self._profile(connection, character_id)
        title = "Fact" if kind == "fact" else "Stm"
        progress[f"pending{title}"] = False
        progress[f"{kind}Rounds"] = 0
        progress[f"{kind}Checkpoint"] = checkpoint
        progress["lastError"] = ""
        connection.execute(
            """
            UPDATE memory_profiles
            SET progress_json = ?, version = version + 1, updated_at = ?
            WHERE character_id = ?
            """,
            (
                json.dumps(progress, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                character_id,
            ),
        )

    def mark_memory_error(self, character_id, message):
        with self.lock, self._connect() as connection:
            row, config, progress = self._profile(connection, character_id)
            progress["lastError"] = _clean_text(message, 1000)
            connection.execute(
                """
                UPDATE memory_profiles
                SET progress_json = ?, version = version + 1, updated_at = ?
                WHERE character_id = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    character_id,
                ),
            )


def load_backend_config() -> dict:
    if os.path.exists(BACKEND_CONFIG_FILE):
        with open(BACKEND_CONFIG_FILE, "r", encoding="utf-8") as file:
            config = json.load(file)
    else:
        config = {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "token": secrets.token_urlsafe(32),
            "allowed_origins": [
                "https://qi3474942281.github.io",
                "http://127.0.0.1:8765",
                "http://localhost:8765",
            ],
        }
        with open(BACKEND_CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
        print(f"Created {BACKEND_CONFIG_FILE}. Keep its token private.")
    return config


def create_api_app(store: ClawBotStore, config: dict, model_loader=None) -> web.Application:
    token = str(config.get("token") or "")
    allowed_origins = set(config.get("allowed_origins") or [])

    @web.middleware
    async def security(request, handler):
        origin = request.headers.get("Origin")
        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            supplied = request.headers.get("Authorization", "")
            if not token or supplied != f"Bearer {token}":
                response = web.json_response({"error": "unauthorized"}, status=401)
            elif origin and origin not in allowed_origins:
                response = web.json_response({"error": "origin not allowed"}, status=403)
            else:
                response = await handler(request)
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
        return response

    app = web.Application(middlewares=[security], client_max_size=2 * 1024 * 1024)

    async def health(_request):
        return web.json_response({"ok": True})

    async def get_state(_request):
        return web.json_response(
            {"state": store.get_state(), "history": store.history_for_web()}
        )

    async def put_state(request):
        try:
            body = await request.json()
            state = store.save_state(body.get("state"))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response({"ok": True, "state": state})

    async def get_memory(request):
        try:
            result = store.memory_snapshot(request.match_info["character_id"])
            result["models"] = model_loader() if model_loader else {}
            return web.json_response(result)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=404)

    async def put_memory_config(request):
        try:
            body = await request.json()
            requested_config = body.get("config")
            models = model_loader() if model_loader else {}
            summary_model = (
                requested_config.get("summaryModel")
                if isinstance(requested_config, dict)
                else ""
            )
            if summary_model and summary_model not in models:
                raise ValueError("总结模型必须来自当前模型列表")
            result = store.save_memory_config(
                request.match_info["character_id"],
                requested_config,
                body.get("version"),
            )
            result["models"] = models
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def post_memory_entry(request):
        try:
            body = await request.json()
            result = store.add_memory_entry(
                request.match_info["character_id"],
                body,
                body.get("version"),
            )
            return web.json_response(result, status=201)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def put_memory_entry(request):
        try:
            body = await request.json()
            result = store.update_memory_entry(
                request.match_info["character_id"],
                request.match_info["entry_id"],
                body,
                body.get("version"),
            )
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def delete_memory_entry(request):
        try:
            body = await request.json()
            store.delete_memory_entry(
                request.match_info["character_id"],
                request.match_info["entry_id"],
                body.get("version"),
            )
            return web.json_response({"ok": True})
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def clear_memory(request):
        try:
            body = await request.json()
            result = store.clear_memory(
                request.match_info["character_id"],
                body.get("version"),
            )
            result["models"] = model_loader() if model_loader else {}
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def delete_message(request):
        try:
            store.delete_message(request.match_info["message_id"])
            return web.json_response(
                {"ok": True, "history": store.history_for_web()}
            )
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def clear_messages(request):
        try:
            history = store.clear_messages(request.match_info["character_id"])
            return web.json_response({"ok": True, "history": history})
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def get_proactive(request):
        try:
            return web.json_response(
                store.proactive_snapshot(request.match_info["character_id"])
            )
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=404)

    async def put_proactive(request):
        try:
            body = await request.json()
            result = store.save_proactive_config(
                request.match_info["character_id"],
                body.get("config"),
                body.get("version"),
            )
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def get_general(request):
        try:
            result = store.general_snapshot(request.match_info["character_id"])
            result["models"] = model_loader() if model_loader else {}
            return web.json_response(result)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=404)

    async def put_general(request):
        try:
            body = await request.json()
            models = model_loader() if model_loader else {}
            result = store.save_general_config(
                request.match_info["character_id"],
                body.get("config"),
                body.get("version"),
                models,
            )
            result["models"] = models
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def get_affection(request):
        try:
            return web.json_response(
                store.affection_snapshot(request.match_info["character_id"])
            )
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=404)

    async def put_affection(request):
        try:
            body = await request.json()
            result = store.save_affection(
                request.match_info["character_id"],
                body.get("value"),
                body.get("version"),
            )
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def reset_affection(request):
        try:
            body = await request.json()
            result = store.reset_affection(
                request.match_info["character_id"],
                body.get("version"),
            )
            return web.json_response(result)
        except VersionConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    app.router.add_get("/api/health", health)
    app.router.add_get("/api/state", get_state)
    app.router.add_put("/api/state", put_state)
    app.router.add_get("/api/memory/{character_id}", get_memory)
    app.router.add_put("/api/memory/{character_id}/config", put_memory_config)
    app.router.add_post("/api/memory/{character_id}/entries", post_memory_entry)
    app.router.add_put(
        "/api/memory/{character_id}/entries/{entry_id}", put_memory_entry
    )
    app.router.add_delete(
        "/api/memory/{character_id}/entries/{entry_id}", delete_memory_entry
    )
    app.router.add_post("/api/memory/{character_id}/clear", clear_memory)
    app.router.add_delete("/api/messages/{message_id}", delete_message)
    app.router.add_post("/api/messages/{character_id}/clear", clear_messages)
    app.router.add_get("/api/proactive/{character_id}", get_proactive)
    app.router.add_put("/api/proactive/{character_id}", put_proactive)
    app.router.add_get("/api/general/{character_id}", get_general)
    app.router.add_put("/api/general/{character_id}", put_general)
    app.router.add_get("/api/affection/{character_id}", get_affection)
    app.router.add_put("/api/affection/{character_id}", put_affection)
    app.router.add_post("/api/affection/{character_id}/reset", reset_affection)
    return app


async def start_backend(store: ClawBotStore, model_loader=None):
    config = load_backend_config()
    app = create_api_app(store, config, model_loader=model_loader)
    runner = web.AppRunner(app)
    await runner.setup()
    host = str(config.get("host") or DEFAULT_HOST)
    port = int(config.get("port") or DEFAULT_PORT)
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"ClawBot backend listening on http://{host}:{port}")
    return runner


def run_backend():
    store = ClawBotStore()

    async def serve():
        await start_backend(store)
        await asyncio.Event().wait()

    asyncio.run(serve())


if __name__ == "__main__":
    run_backend()
