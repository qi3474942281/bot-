import io
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


REPO_OWNER = "qi3474942281"
REPO_NAME = "bot-"
REPO_BRANCH = "main"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{REPO_BRANCH}"
SETTINGS_URL = f"{RAW_BASE_URL}/settings.json"
ARCHIVE_URL = f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/zip/refs/heads/{REPO_BRANCH}"

CONFIG_FILE = "config.json"
PUBLIC_KEYS = ("default_model", "current_model", "models")
NEUTRAL_PROMPT = (
    "你是一个正在扮演指定人物的聊天 AI。必须优先遵守后端传入的人设、"
    "规则、记忆和聊天风格，不要自称其他助手。"
)
FIXED_IDENTITY_TERMS = ("Kiro", "kiro", "代码助手", "编程助手")

# Files that belong to the app itself and are safe to replace from GitHub.
SYNC_FILES = {
    ".gitignore",
    "README.md",
    "backend_config.example.json",
    "bot.js",
    "bot.py",
    "clawbot_backend.py",
    "config.example.json",
    "deepseek.py",
    "dusapi.py",
    "index.html",
    "settings.json",
    "sync_from_github.py",
    "weixin-bot-api.md",
    "weixin-openclaw-api-py-docs.md",
}
SYNC_DIRS = {"docs", "tests", "deploy", ".github"}

# Local secrets/state/tools. These must never be replaced by a GitHub sync.
PROTECTED_NAMES = {
    CONFIG_FILE,
    "backend_config.json",
    "clawbot-data.sqlite3",
    "cloudflared.exe",
    "qrcode.png",
    "sync-and-start.cmd",
    "start-backend-only.cmd",
    "start-cloudflare-tunnel.cmd",
    "start-cloudflare-tunnel-log.cmd",
    "start-server-backend.cmd",
}
PROTECTED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".key", ".pem"}
BACKUP_ROOT = ".cloud-sync"


def _request(url: str):
    return urllib.request.Request(
        url,
        headers={"User-Agent": "weixin-ClawBot-API-github-sync"},
    )


def _normalized_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _is_safe_relative_path(path: str) -> bool:
    normalized = _normalized_path(path)
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        return False
    return not Path(normalized).is_absolute()


def should_sync_path(path: str) -> bool:
    normalized = _normalized_path(path)
    if not _is_safe_relative_path(normalized):
        return False
    name = Path(normalized).name
    if name in PROTECTED_NAMES or Path(normalized).suffix.lower() in PROTECTED_SUFFIXES:
        return False
    if normalized in SYNC_FILES:
        return True
    top_level = normalized.split("/", 1)[0]
    return top_level in SYNC_DIRS


def _backup_existing(target: Path, backup_dir: Path):
    if not target.exists():
        return
    relative = target.relative_to(Path.cwd())
    backup_target = backup_dir / relative
    backup_target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        if backup_target.exists():
            shutil.rmtree(backup_target)
        shutil.copytree(target, backup_target)
    else:
        shutil.copy2(target, backup_target)


def _copy_file(source: Path, target: Path, backup_dir: Path):
    _backup_existing(target, backup_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _replace_directory(source: Path, target: Path, backup_dir: Path):
    _backup_existing(target, backup_dir)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _download_archive() -> bytes:
    with urllib.request.urlopen(_request(ARCHIVE_URL), timeout=30) as response:
        return response.read()


def _contains_fixed_identity_prompt(value) -> bool:
    text = str(value or "")
    return any(term in text for term in FIXED_IDENTITY_TERMS)


def sanitize_fixed_identity_prompts(config: dict) -> int:
    changed = 0
    providers = config.get("providers")
    if isinstance(providers, dict):
        for provider in providers.values():
            if not isinstance(provider, dict):
                continue
            if _contains_fixed_identity_prompt(provider.get("prompt")):
                provider["prompt"] = NEUTRAL_PROMPT
                changed += 1

    models = config.get("models")
    if isinstance(models, dict):
        for model in models.values():
            if not isinstance(model, dict):
                continue
            if _contains_fixed_identity_prompt(model.get("prompt")):
                model["prompt"] = NEUTRAL_PROMPT
                changed += 1

    if _contains_fixed_identity_prompt(config.get("prompt")):
        config["prompt"] = NEUTRAL_PROMPT
        changed += 1
    return changed


def sync_code_from_github() -> bool:
    try:
        archive = _download_archive()
        with tempfile.TemporaryDirectory(prefix="clawbot-sync-") as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
                zip_file.extractall(temp_path)

            roots = [item for item in temp_path.iterdir() if item.is_dir()]
            if not roots:
                raise ValueError("GitHub archive did not contain a project folder")
            source_root = roots[0]
            backup_dir = (
                Path.cwd()
                / BACKUP_ROOT
                / datetime.now().strftime("%Y%m%d-%H%M%S")
            )

            synced = 0
            for source in source_root.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(source_root).as_posix()
                if relative.split("/", 1)[0] in SYNC_DIRS:
                    continue
                if not should_sync_path(relative):
                    continue
                target = Path.cwd() / relative
                _copy_file(source, target, backup_dir)
                synced += 1

            for dirname in sorted(SYNC_DIRS):
                source = source_root / dirname
                if not source.is_dir():
                    continue
                target = Path.cwd() / dirname
                _replace_directory(source, target, backup_dir)

            print(
                "Program files synchronized from GitHub. "
                f"Updated {synced} files; backup: {backup_dir}"
            )
            return True
    except Exception as error:
        print(f"GitHub program sync skipped: {error}")
        return False


def sync_public_settings() -> bool:
    if not os.path.exists(CONFIG_FILE):
        print("config.json not found; skipping public settings sync.")
        return False

    try:
        with urllib.request.urlopen(_request(SETTINGS_URL), timeout=10) as response:
            remote = json.load(response)

        with open(CONFIG_FILE, "r", encoding="utf-8-sig") as file:
            local = json.load(file)

        models = remote.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("settings.json must contain a non-empty models object")

        for key in ("default_model", "current_model"):
            if remote.get(key) not in models:
                raise ValueError(f"{key} must reference a key in models")

        for key in PUBLIC_KEYS:
            local[key] = remote[key]

        cleaned = sanitize_fixed_identity_prompts(local)

        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(local, file, ensure_ascii=False, indent=2)

        print("Public settings synchronized from GitHub.")
        if cleaned:
            print(f"Cleaned {cleaned} fixed identity prompt(s) in config.json.")
        return True
    except Exception as error:
        print(f"GitHub settings sync skipped: {error}")
        return False


def main():
    sync_code_from_github()
    sync_public_settings()


if __name__ == "__main__":
    main()
