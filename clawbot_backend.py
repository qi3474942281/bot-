import asyncio
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone

from aiohttp import web


DB_FILE = "clawbot-data.sqlite3"
BACKEND_CONFIG_FILE = "backend_config.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _default_state() -> dict:
    character_id = "default"
    return {
        "activeCharacterId": character_id,
        "characters": [
            {
                "id": character_id,
                "name": "默认 ClawBot",
                "persona": {
                    "description": "",
                    "traits": "",
                    "examples": "",
                },
                "memories": [],
                "promptMemories": [],
            }
        ],
    }


def _clean_text(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            existing = connection.execute(
                "SELECT 1 FROM settings WHERE key = 'state'"
            ).fetchone()
            if not existing:
                connection.execute(
                    "INSERT INTO settings(key, value) VALUES('state', ?)",
                    (json.dumps(_default_state(), ensure_ascii=False),),
                )

    def get_state(self) -> dict:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = 'state'"
            ).fetchone()
        return _normalize_state(json.loads(row["value"]))

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
        return state

    def active_character(self) -> dict:
        state = self.get_state()
        return next(
            item
            for item in state["characters"]
            if item["id"] == state["activeCharacterId"]
        )

    def build_prompt(self, fallback_prompt: str) -> str:
        character = self.active_character()
        persona = character["persona"]
        sections = [fallback_prompt, f"你当前扮演：{character['name']}。"]
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
        if character["memories"]:
            sections.append(
                "【需要记住的信息】\n"
                + "\n".join(f"- {item}" for item in character["memories"])
            )
        return "\n\n".join(section for section in sections if section)

    def add_message(self, user_id: str, role: str, content: str):
        character = self.active_character()
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(character_id, user_id, role, content, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    character["id"],
                    _clean_text(user_id, 200),
                    role,
                    _clean_text(content, 20000),
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

    def recent_history(self, user_id: str, limit: int = 20) -> list[dict]:
        character = self.active_character()
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE character_id = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (character["id"], user_id, max(1, min(limit, 100))),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
        ]

    def history_for_web(self, limit: int = 300) -> list[dict]:
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, character_id, user_id, role, content, created_at
                FROM messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [dict(row) for row in rows]


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


def create_api_app(store: ClawBotStore, config: dict) -> web.Application:
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
                response = web.json_response(
                    {"error": "origin not allowed"}, status=403
                )
            else:
                response = await handler(request)
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = (
                "Authorization, Content-Type"
            )
            response.headers["Access-Control-Allow-Methods"] = "GET, PUT, OPTIONS"
        return response

    app = web.Application(middlewares=[security], client_max_size=2 * 1024 * 1024)

    async def health(_request):
        return web.json_response({"ok": True})

    async def get_state(_request):
        return web.json_response(
            {
                "state": store.get_state(),
                "history": store.history_for_web(),
            }
        )

    async def put_state(request):
        try:
            body = await request.json()
            state = store.save_state(body.get("state"))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response({"ok": True, "state": state})

    app.router.add_get("/api/health", health)
    app.router.add_get("/api/state", get_state)
    app.router.add_put("/api/state", put_state)
    return app


async def start_backend(store: ClawBotStore):
    config = load_backend_config()
    app = create_api_app(store, config)
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
