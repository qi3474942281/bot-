import json
import os
import urllib.request


SETTINGS_URL = (
    "https://raw.githubusercontent.com/qi3474942281/bot-/main/settings.json"
)
CONFIG_FILE = "config.json"
PUBLIC_KEYS = ("default_model", "current_model", "models")


def sync_public_settings() -> bool:
    if not os.path.exists(CONFIG_FILE):
        print("config.json not found; skipping public settings sync.")
        return False

    try:
        request = urllib.request.Request(
            SETTINGS_URL,
            headers={"User-Agent": "weixin-ClawBot-API-settings-sync"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
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

        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(local, file, ensure_ascii=False, indent=2)

        print("Public settings synchronized from GitHub.")
        return True
    except Exception as error:
        print(f"GitHub settings sync skipped: {error}")
        return False


if __name__ == "__main__":
    sync_public_settings()

