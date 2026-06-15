import asyncio
import base64
import io
import json
import os
import random
import re
import aiohttp
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from dusapi import DusAPI, DusConfig
from deepseek import DeepSeekAPI, DeepSeekConfig
from sync_from_github import sync_public_settings
from clawbot_backend import (
    ClawBotStore,
    defer_to_proactive_window,
    proactive_allowed_at,
    start_backend,
)

executor = ThreadPoolExecutor(max_workers=4)
ai = None  # 启动时从配置文件加载后初始化
clawbot_store = ClawBotStore()

# ========== 自动重连配置（可调参数） ==========
# 测试时将数值改小，例如：
#   "session_duration": 300, "warning_before": 60, "reminder_interval": 30,
#   "force_before": 60, "qrcode_scan_timeout": 120
RECONNECT_CONFIG = {
    "session_duration":    24 * 3600,  # 会话总时长（秒）
    "warning_before":       2 * 3600,  # 提前多久发出警告（秒）
    "reminder_interval":      30 * 60, # 用户回 N 后多久再问（秒）
    "force_before":           30 * 60, # 最后多久强制重连（秒）
    "qrcode_scan_timeout":       600,  # 等待用户扫码最长时间（秒）
}
# =============================================

# ========== 配置文件 ==========
CONFIG_FILE = "config.json"
_DEFAULT_PROMPT = "你是一个有帮助的AI助手，请用中文简洁地回复。字数尽量少一些"
CHANNEL_VERSION = "2.4.3"
ILINK_APP_ID = "bot"
ILINK_APP_CLIENT_VERSION = str((2 << 16) | (4 << 8) | 3)
BOT_AGENT = "weixin-ClawBot-API/1.0.1 (python)"

# Only public model preferences are synchronized. Secrets remain in config.json.
sync_public_settings()

PROVIDERS = {
    "dusapi": {
        "label": "DusAPI",
        "base_url": "https://api.dusapi.com",
        "model": "gpt-5",
        "prompt": _DEFAULT_PROMPT,
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "prompt": _DEFAULT_PROMPT,
    },
    "packy": {
        "label": "Packy API",
        "base_url": "https://www.packyapi.com/v1",
        "model": "claude-opus-4-8",
        "prompt": _DEFAULT_PROMPT,
    },
}


def get_model_profiles() -> dict:
    cfg = load_config_file()
    return cfg.get("models") or {}


def get_active_model_key() -> str | None:
    cfg = load_config_file()
    models = cfg.get("models") or {}
    if not models:
        return None
    preferred = cfg.get("current_model") or cfg.get("default_model")
    if preferred in models:
        return preferred
    return next(iter(models))


def get_active_model_name() -> str | None:
    models = get_model_profiles()
    active_key = get_active_model_key()
    if not active_key:
        return None
    return models.get(active_key, {}).get("model")


def get_character_model_key(character_id: str) -> str | None:
    models = get_model_profiles()
    configured = clawbot_store.general_snapshot(character_id)["config"]["currentModel"]
    if configured in models:
        return configured
    return get_active_model_key()


def resolve_model_runtime(model_key: str | None = None) -> dict:
    cfg = load_config_file()
    models = cfg.get("models") or {}
    if model_key is None:
        model_key = get_active_model_key()
    profile = models.get(model_key or "", {}) if isinstance(models, dict) else {}
    provider = profile.get("provider") or cfg.get("provider") or "dusapi"
    provider_cfg = dict((cfg.get("providers") or {}).get(provider) or {})
    defaults = PROVIDERS.get(provider, PROVIDERS["dusapi"])

    return {
        "provider": provider,
        "api_key": profile.get("api_key") or provider_cfg.get("api_key", ""),
        "base_url": profile.get("base_url") or provider_cfg.get("base_url") or defaults["base_url"],
        "model": profile.get("model") or provider_cfg.get("model") or defaults["model"],
        "prompt": profile.get("prompt") or provider_cfg.get("prompt") or defaults["prompt"],
    }


def create_ai_client(runtime: dict):
    provider = runtime.get("provider", "dusapi")
    if provider == "dusapi":
        return DusAPI(DusConfig(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            model1=runtime["model"],
            prompt=runtime["prompt"],
        ))
    return DeepSeekAPI(DeepSeekConfig(
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        model=runtime["model"],
        prompt=runtime["prompt"],
    ))


def save_character_model_key(character_id: str, model_key: str):
    snapshot = clawbot_store.general_snapshot(character_id)
    config = dict(snapshot["config"])
    config["currentModel"] = model_key
    clawbot_store.save_general_config(
        character_id, config, snapshot["version"], get_model_profiles()
    )


def format_model_list() -> str:
    models = get_model_profiles()
    character_id = clawbot_store.active_character()["id"]
    active_key = get_character_model_key(character_id)
    if not models:
        return "No models configured."

    lines = ["Available models:"]
    for key, item in models.items():
        mark = " *" if key == active_key else ""
        lines.append(f"/model {key}{mark} -> {item.get('model', '')}")
    return "\n".join(lines)


def handle_model_command(text: str) -> tuple[bool, str | None]:
    stripped = text.strip()
    if not stripped.startswith("/model"):
        return False, None

    parts = stripped.split(maxsplit=1)
    if len(parts) == 1:
        return True, format_model_list()

    model_key = parts[1].strip().lower()
    models = get_model_profiles()
    if model_key not in models:
        return True, f"Unknown model: {model_key}\n\n{format_model_list()}"

    character_id = clawbot_store.active_character()["id"]
    save_character_model_key(character_id, model_key)
    model_name = models[model_key].get("model", "")
    return True, f"Switched model to {model_key}: {model_name}"


def format_memory_settings() -> str:
    character = clawbot_store.active_character()
    snapshot = clawbot_store.memory_snapshot(character["id"])
    config = snapshot["config"]
    progress = snapshot["progress"]
    model = (
        config["summaryModel"]
        or get_character_model_key(character["id"])
        or "未配置"
    )
    return "\n".join(
        [
            "记忆设置：",
            f"- 总结模型：{model}",
            f"- 关键事实：每 {config['factRounds']} 轮，自动生成 {'开' if config['factsAuto'] else '关'}",
            f"- 短期记忆：每 {config['stmRounds']} 轮，{config['stmMinChars']}–{config['stmMaxChars']} 字",
            f"- 长期记忆：每 {config['ltmAfterStm']} 条 STM，{config['ltmMinChars']}–{config['ltmMaxChars']} 字",
            f"- STM/LTM 引擎：{'开' if config['engineAuto'] else '关'}",
            f"- 当前进度：事实 {progress['factRounds']}/{config['factRounds']}，STM {progress['stmRounds']}/{config['stmRounds']}",
            "",
            "命令：/memory set <项目> <值>",
            "项目：fact-rounds、stm-rounds、stm-min、stm-max、ltm-after-stm、ltm-min、ltm-max、model、facts-auto、engine-auto",
        ]
    )


def handle_memory_command(text: str) -> tuple[bool, str | None]:
    stripped = text.strip()
    if not stripped.startswith("/memory"):
        return False, None
    if stripped == "/memory":
        return True, format_memory_settings()

    parts = stripped.split()
    if len(parts) != 4 or parts[1].lower() != "set":
        return True, "格式错误。\n\n" + format_memory_settings()

    key, raw_value = parts[2].lower(), parts[3]
    field_map = {
        "fact-rounds": "factRounds",
        "stm-rounds": "stmRounds",
        "stm-min": "stmMinChars",
        "stm-max": "stmMaxChars",
        "ltm-after-stm": "ltmAfterStm",
        "ltm-min": "ltmMinChars",
        "ltm-max": "ltmMaxChars",
    }
    character = clawbot_store.active_character()
    snapshot = clawbot_store.memory_snapshot(character["id"])
    config = dict(snapshot["config"])
    try:
        if key in field_map:
            value = int(raw_value)
            if value <= 0:
                raise ValueError("数值必须大于 0")
            config[field_map[key]] = value
        elif key == "model":
            if raw_value not in get_model_profiles():
                raise ValueError(f"未知模型：{raw_value}")
            config["summaryModel"] = raw_value
        elif key in {"facts-auto", "engine-auto"}:
            if raw_value.lower() not in {"on", "off"}:
                raise ValueError("开关值只能是 on 或 off")
            config["factsAuto" if key == "facts-auto" else "engineAuto"] = (
                raw_value.lower() == "on"
            )
        else:
            raise ValueError(f"未知设置项：{key}")
        clawbot_store.save_memory_config(
            character["id"], config, snapshot["version"]
        )
        return True, "记忆设置已更新。\n\n" + format_memory_settings()
    except (TypeError, ValueError) as error:
        return True, f"记忆设置未修改：{error}"


def _conversation_text(messages):
    labels = {"user": "用户", "assistant": "ClawBot"}
    return "\n".join(
        f"[{item['created_at']}] {labels.get(item['role'], item['role'])}：{item['content']}"
        for item in messages
    )


def _extract_json(text):
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", value, re.I)
    if fenced:
        value = fenced.group(1).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("总结模型未返回 JSON")
    result = json.loads(value[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("总结结果必须是对象")
    return result


def split_reply_messages(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= 50:
        return [value]

    tokens = re.split(
        r"(```[\s\S]*?```|https?://\S+|[^\n。！？!?；;]+[。！？!?；;]?|\n+)",
        value,
    )
    units = [item.strip() for item in tokens if item and item.strip()]
    target = 60 if len(value) <= 120 else 85
    parts = []
    current = ""
    for unit in units:
        if unit.startswith("```") or unit.startswith(("http://", "https://")):
            if current:
                parts.append(current)
                current = ""
            parts.append(unit)
            continue
        candidate = f"{current}{unit}" if current else unit
        if current and len(candidate) > target:
            parts.append(current)
            current = unit
        else:
            current = candidate
    if current:
        parts.append(current)

    merged = []
    for part in parts:
        if merged and len(part) < 18 and not part.startswith(("```", "http")):
            merged[-1] += part
        else:
            merged.append(part)
    if len(value) <= 120 and len(merged) > 2:
        midpoint = max(1, len(merged) // 2)
        merged = ["".join(merged[:midpoint]), "".join(merged[midpoint:])]
    return [item for item in merged if item]


def _chat_result_prompt(character: dict, repair_error=None) -> str:
    rules = "\n".join(f"- {item}" for item in character.get("promptMemories", []))
    repair = (
        f"\n上一次输出格式不合格：{repair_error}。请修正并只输出 JSON。"
        if repair_error
        else ""
    )
    return (
        "\n\n【本轮输出格式】\n"
        "只输出一个 JSON 对象，不要使用 Markdown 代码块："
        '{"reply":"给用户的自然回复正文","affectionDelta":整数,'
        '"affectionReason":"简短原因","ruleOverride":true或false}。\n'
        "reply 必须符合人物设定，可以包含多句和自然分段。\n"
        "无明确好感度数值规则时，affectionDelta 必须在 -5 到 5 之间；"
        "只有下列必须遵守的规则明确写出了数值，并且本轮确实命中时，"
        "ruleOverride 才能为 true，并按规则返回增减值。\n"
        f"必须遵守的规则：\n{rules or '无明确好感度数值规则'}"
        f"{repair}"
    )


def generate_chat_result(
    text: str, character_id: str, model_key: str, user_id: str
) -> dict:
    runtime = resolve_model_runtime(model_key)
    chat_ai = create_ai_client(runtime)
    character = clawbot_store.character(character_id)
    base_prompt = clawbot_store.build_prompt(runtime["prompt"], character_id)
    history = clawbot_store.recent_history(
        user_id, character_id=character_id
    )
    last_error = None
    fallback_text = ""
    for _attempt in range(2):
        raw = chat_ai.chat(
            text,
            model=runtime["model"],
            prompt=base_prompt + _chat_result_prompt(character, last_error),
            history=history,
        )
        fallback_text = str(raw or "").strip() or fallback_text
        try:
            result = _extract_json(raw)
            reply = str(result.get("reply") or "").strip()
            if not reply:
                raise ValueError("reply 不能为空")
            delta = int(result.get("affectionDelta", 0))
            rule_override = bool(result.get("ruleOverride", False))
            if not rule_override:
                delta = max(-5, min(5, delta))
            else:
                delta = max(-500, min(500, delta))
            return {
                "reply": reply,
                "affectionDelta": delta,
                "affectionReason": str(
                    result.get("affectionReason") or ""
                ).strip()[:1000],
            }
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = str(error)
    fallback_reply = fallback_text
    try:
        fallback_result = _extract_json(fallback_text)
        if isinstance(fallback_result, dict) and fallback_result.get("reply"):
            fallback_reply = str(fallback_result["reply"]).strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "reply": fallback_reply,
        "affectionDelta": 0,
        "affectionReason": "",
    }


def _summary_prompt(task, repair_error=None):
    config = task["config"]
    repair = (
        f"\n上一次输出不合格：{repair_error}。请修正并只输出 JSON。"
        if repair_error
        else ""
    )
    if task["kind"] == "fact":
        facts = "\n".join(
            f"- id={item['id']}: {item['content']}"
            for item in task["existingAiFacts"]
        ) or "无"
        return (
            "从对话中提取一条必须长期记住的关键事实。每次必须产生结果。"
            "若与下列 AI 事实相近，可融合并返回 merge；不得融合或改写用户手动事实。"
            "内容应为一句或简短几句，保留专有名称、称呼、承诺和约定。\n"
            f"已有 AI 事实：\n{facts}\n"
            '只输出 JSON：{"action":"add","content":"..."} 或 '
            '{"action":"merge","entryId":数字,"content":"融合后的内容"}。\n'
            f"对话：\n{_conversation_text(task['messages'])}{repair}"
        )
    if task["kind"] == "stm":
        return (
            "将以下对话总结为一条客观短期记忆。保留时间顺序、专有名称、特殊称呼、"
            "重要约定、承诺和宣言，不主观评价或推演。"
            f"content 必须为 {config['stmMinChars']}–{config['stmMaxChars']} 个中文字符左右。"
            '只输出 JSON：{"timeLabel":"时间或时间范围","scene":"场景",'
            '"content":"客观事件"}。\n'
            f"对话：\n{_conversation_text(task['messages'])}{repair}"
        )
    stm_text = "\n".join(
        f"{item['sequence_no']}. {item['time_label']} | {item['scene']} | {item['content']}"
        for item in task["stm"]
    )
    return (
        "严格基于全部 STM 按先后顺序生成一条长期记忆。每条 STM 的信息占比尽量均衡，"
        "不得遗漏剧情发展和角色细节，不得主观评价、推演未来或添加总结性结论。"
        f"content 必须为 {config['ltmMinChars']}–{config['ltmMaxChars']} 个中文字符左右。"
        '只输出 JSON：{"timeLabel":"整体时间范围","scene":"场景范围",'
        '"content":"长期记忆正文"}。\n'
        f"STM：\n{stm_text}{repair}"
    )


def _validate_summary_result(task, result):
    content = str(result.get("content") or "").strip()
    if not content:
        raise ValueError("content 不能为空")
    if task["kind"] == "fact":
        if result.get("action") not in {"add", "merge"}:
            raise ValueError("action 必须是 add 或 merge")
        if result.get("action") == "merge":
            allowed_ids = {item["id"] for item in task["existingAiFacts"]}
            if result.get("entryId") not in allowed_ids:
                raise ValueError("merge 只能引用已有 AI 事实的 entryId")
        return
    minimum = task["config"]["stmMinChars" if task["kind"] == "stm" else "ltmMinChars"]
    maximum = task["config"]["stmMaxChars" if task["kind"] == "stm" else "ltmMaxChars"]
    if not minimum <= len(content) <= maximum:
        raise ValueError(f"content 字数必须在 {minimum}–{maximum} 之间，当前为 {len(content)}")
    if not str(result.get("timeLabel") or "").strip():
        raise ValueError("timeLabel 不能为空")
    if not str(result.get("scene") or "").strip():
        raise ValueError("scene 不能为空")


def run_memory_summary(task):
    model_key = task["config"]["summaryModel"] or get_character_model_key(
        task["characterId"]
    )
    profiles = get_model_profiles()
    if model_key not in profiles:
        model_key = get_character_model_key(task["characterId"])
    runtime = resolve_model_runtime(model_key)
    summary_ai = create_ai_client(runtime)
    last_error = None
    for attempt in range(2):
        prompt = _summary_prompt(task, last_error)
        raw = summary_ai.chat(
            prompt,
            model=runtime["model"],
            prompt="你是 ClawBot 的结构化记忆引擎。严格遵循用户要求，只输出指定 JSON。",
            history=None,
        )
        try:
            result = _extract_json(raw)
            _validate_summary_result(task, result)
            return result
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = str(error)
    raise ValueError(last_error or "记忆总结失败")


async def process_memory_tasks(loop, character_id, user_id):
    failed = False
    tasks = clawbot_store.prepare_memory_tasks(character_id, user_id)
    for task in tasks:
        try:
            result = await loop.run_in_executor(
                executor, partial(run_memory_summary, task)
            )
            if task["kind"] == "fact":
                clawbot_store.complete_fact_task(task, result)
            elif task["kind"] == "stm":
                needs_ltm = clawbot_store.complete_stm_task(task, result)
                if needs_ltm:
                    ltm_task = clawbot_store.prepare_ltm_task(character_id)
                    ltm_result = await loop.run_in_executor(
                        executor, partial(run_memory_summary, ltm_task)
                    )
                    clawbot_store.complete_ltm_task(ltm_task, ltm_result)
            else:
                clawbot_store.complete_ltm_task(task, result)
        except Exception as error:
            failed = True
            clawbot_store.mark_memory_error(character_id, str(error))
            print(f"记忆总结失败: {error}")
    return failed


def mask_key(key: str) -> str:
    """保留前5位和后5位，中间用星号替换。"""
    if len(key) <= 10:
        return key
    return key[:5] + "*" * (len(key) - 10) + key[-5:]


def load_config_file() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {"provider": "dusapi", "providers": {}}

    with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    # 兼容旧版扁平配置：{api_key, base_url, model, prompt}
    if "providers" not in cfg:
        old_provider_cfg = {
            "api_key": cfg.get("api_key", ""),
            "base_url": cfg.get("base_url", PROVIDERS["dusapi"]["base_url"]),
            "model": cfg.get("model", PROVIDERS["dusapi"]["model"]),
            "prompt": cfg.get("prompt", _DEFAULT_PROMPT),
        }
        cfg = {
            "provider": "dusapi",
            "providers": {"dusapi": old_provider_cfg},
        }
    cfg.setdefault("provider", "dusapi")
    cfg.setdefault("providers", {})
    return cfg


def save_config_file(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def choose_provider(default_provider: str) -> str:
    print("\n请选择 AI 提供商：")
    keys = list(PROVIDERS.keys())
    for index, key in enumerate(keys, 1):
        default_mark = "（默认）" if key == default_provider else ""
        print(f"  {index}. {PROVIDERS[key]['label']} {default_mark}")

    while True:
        choice = input("输入序号或名称后回车: ").strip().lower()
        if not choice:
            return default_provider if default_provider in PROVIDERS else "dusapi"
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        if choice in PROVIDERS:
            return choice
        print("输入无效，请重新选择。")


def prompt_provider_config(provider: str, old_cfg: dict | None = None) -> dict:
    defaults = PROVIDERS[provider]
    old_cfg = old_cfg or {}
    print(f"\n配置 {defaults['label']}：")

    old_key = old_cfg.get("api_key", "")
    key_prompt = f"请输入 API Key（当前 {mask_key(old_key)}，留空沿用）: " if old_key else "请输入 API Key: "
    api_key = input(key_prompt).strip() or old_key

    old_base_url = old_cfg.get("base_url", defaults["base_url"])
    base_url = input(f"请输入 API 地址（留空默认/沿用 {old_base_url}）: ").strip() or old_base_url

    old_model = old_cfg.get("model", defaults["model"])
    model = input(f"请输入模型名称（留空默认/沿用 {old_model}）: ").strip() or old_model

    old_prompt = old_cfg.get("prompt", defaults["prompt"])
    prompt = input("请输入系统提示词（留空默认/沿用当前值）: ").strip() or old_prompt

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
    }


def load_or_create_config() -> dict:
    """先选择 AI 提供商，再确认或创建对应配置。"""
    sep = "=" * 60
    dash = "-" * 60
    cfg = load_config_file()

    while True:
        provider = choose_provider(cfg.get("provider", "dusapi"))
        cfg["provider"] = provider
        provider_cfg = cfg["providers"].get(provider)
        label = PROVIDERS[provider]["label"]

        if not provider_cfg:
            print(f"\n未找到 {label} 配置，需要创建。")
            provider_cfg = prompt_provider_config(provider)
            cfg["providers"][provider] = provider_cfg
            save_config_file(cfg)
            print(f"\n配置已保存到 {CONFIG_FILE}\n")
            return {"provider": provider, **provider_cfg}

        print(f"\n{sep}")
        print(f"  当前选择：{label}")
        print("  当前配置如下：")
        print(sep)
        print(f"  API Key  : {mask_key(provider_cfg.get('api_key', ''))}")
        print(f"  API 地址 : {provider_cfg.get('base_url', '')}")
        print(f"  模型     : {provider_cfg.get('model', '')}")
        prompt_preview = provider_cfg.get("prompt", "")[:50]
        print(f"  提示词   : {prompt_preview}{'...' if len(provider_cfg.get('prompt','')) > 50 else ''}")
        print(dash)

        choice = input("\n使用此配置继续？(直接回车或输入 Y 继续 / 输入 N 重新配置 / 输入 S 切换提供商): ").strip().upper()
        if choice == "N":
            provider_cfg = prompt_provider_config(provider, provider_cfg)
            cfg["providers"][provider] = provider_cfg
            save_config_file(cfg)
            print(f"\n配置已保存到 {CONFIG_FILE}\n")
            return {"provider": provider, **provider_cfg}
        if choice == "S":
            continue
        else:
            save_config_file(cfg)
            return {"provider": provider, **provider_cfg}
# ==============================

BASE_URL = "https://ilinkai.weixin.qq.com"
COMMANDS_MSG = (
    "连接成功！\n"
    "可用指令：\n"
    "/help  /指令   - 查看全部指令列表\n"
    "/time          - 查询当前连接剩余时间\n"
    "/重新连接       - 立即触发重新连接（需确认）\n"
    "\n非指令输入即为 AI 对话"
)


def make_headers(token=None):
    uin = str(random.randint(0, 0xFFFFFFFF))
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": base64.b64encode(uin.encode()).decode(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": ILINK_APP_CLIENT_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def base_info():
    return {
        "channel_version": CHANNEL_VERSION,
        "bot_agent": BOT_AGENT,
    }


async def api_get(session, path, token=None, base_url=None):
    url = f"{base_url or BASE_URL}/{path}"
    async with session.get(url, headers=make_headers(token)) as res:
        text = await res.text()
        print(f"  [GET {path}] HTTP {res.status} → {text[:200]}")
        try:
            return json.loads(text)
        except Exception:
            return {}


async def api_post(session, path, body, token=None, base_url=None):
    url = f"{base_url or BASE_URL}/{path}"
    async with session.post(url, json=body, headers=make_headers(token)) as res:
        text = await res.text()
        print(f"  [{path}] HTTP {res.status} → {text[:200]}")
        try:
            import json
            return json.loads(text)
        except Exception:
            return {}


async def send_msg_safe(session, to_id, context_token, text, bot_token_ref, bot_base_url_ref):
    """发送微信消息，失败时降级为控制台打印，不抛异常。"""
    if not to_id or not context_token:
        print(f"[重连通知] {text}")
        return
    try:
        client_id = f"openclaw-weixin-{random.randint(0, 0xFFFFFFFF):08x}"
        await api_post(
            session,
            "ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": to_id,
                    "client_id": client_id,
                    "message_type": 2,
                    "message_state": 2,
                    "context_token": context_token,
                    "item_list": [{"type": 1, "text_item": {"text": text}}],
                },
                "base_info": base_info(),
            },
            bot_token_ref[0],
            bot_base_url_ref[0] or None,
        )
    except Exception as e:
        print(f"[重连通知] 发送失败({e})，降级打印: {text}")


async def send_text_message(
    session, to_id, context_token, text, bot_token_ref, bot_base_url_ref
):
    client_id = f"openclaw-weixin-{random.randint(0, 0xFFFFFFFF):08x}"
    body = {
        "msg": {
            "from_user_id": "",
            "to_user_id": to_id,
            "client_id": client_id,
            "message_type": 2,
            "message_state": 2,
            "context_token": context_token,
            "item_list": [{"type": 1, "text_item": {"text": text}}],
        },
        "base_info": base_info(),
    }
    url = f"{bot_base_url_ref[0] or BASE_URL}/ilink/bot/sendmessage"
    async with session.post(
        url,
        json=body,
        headers=make_headers(bot_token_ref[0]),
    ) as response:
        raw = await response.text()
        if response.status >= 400:
            raise RuntimeError(
                f"WeChat HTTP {response.status}: {raw[:200]}"
            )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError("WeChat returned an invalid response") from error
    if isinstance(result, dict) and result.get("errcode") not in (None, 0):
        raise RuntimeError(
            result.get("errmsg") or f"WeChat API error {result['errcode']}"
        )
    return result


async def send_reply_parts(
    session, to_id, context_token, parts, bot_token_ref, bot_base_url_ref
):
    for index, part in enumerate(parts):
        await send_text_message(
            session,
            to_id,
            context_token,
            part,
            bot_token_ref,
            bot_base_url_ref,
        )
        if index + 1 < len(parts):
            await asyncio.sleep(random.uniform(1, 5))


PROACTIVE_INTERVALS = {
    "random_1_30": (1, 30),
    "random_30_60": (30, 60),
    "random_60_120": (60, 120),
    "random_120_180": (120, 180),
    "random_180_240": (180, 240),
    "random_240_300": (240, 300),
    "random_1_300": (1, 300),
}


def next_proactive_time(config, now=None):
    now = now or datetime.now(timezone.utc)
    if config["mode"] == "exact":
        minutes = config["exactMinutes"]
    else:
        minimum, maximum = PROACTIVE_INTERVALS.get(
            config["mode"], PROACTIVE_INTERVALS["random_60_120"]
        )
        minutes = random.randint(minimum, maximum)
    return defer_to_proactive_window(now + timedelta(minutes=minutes), config)


def generate_proactive_message(user_id, character_id):
    runtime = resolve_model_runtime(get_character_model_key(character_id))
    proactive_ai = create_ai_client(runtime)
    prompt = clawbot_store.build_prompt(runtime["prompt"], character_id)
    prompt += (
        "\n\n【主动消息要求】\n"
        "现在请你自然地主动开启一段对话。结合人物设定、记忆和最近聊天，"
        "发送一条简短、具体、像真人自然想起对方时会说的话。"
        "不要提及定时、系统、任务、频率或主动消息设置，不要重复最近已经说过的话。"
    )
    return proactive_ai.chat(
        "请根据当前关系和上下文，生成一条现在适合主动发给用户的消息。",
        model=runtime["model"],
        prompt=prompt,
        history=clawbot_store.recent_history(
            user_id, character_id=character_id
        ),
    )


async def send_proactive_message(
    session, to_id, context_token, text, bot_token_ref, bot_base_url_ref
):
    client_id = f"openclaw-weixin-{random.randint(0, 0xFFFFFFFF):08x}"
    result = await api_post(
        session,
        "ilink/bot/sendmessage",
        {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_id,
                "client_id": client_id,
                "message_type": 2,
                "message_state": 2,
                "context_token": context_token,
                "item_list": [{"type": 1, "text_item": {"text": text}}],
            },
            "base_info": base_info(),
        },
        bot_token_ref[0],
        bot_base_url_ref[0] or None,
    )
    if isinstance(result, dict) and result.get("errcode") not in (None, 0):
        raise RuntimeError(result.get("errmsg") or f"微信接口错误 {result['errcode']}")


async def proactive_message_task(
    session, bot_token_ref, bot_base_url_ref, last_contact
):
    while True:
        await asyncio.sleep(10)
        try:
            character_id = clawbot_store.active_character()["id"]
            snapshot = clawbot_store.proactive_snapshot(character_id)
            config = snapshot["config"]
            if not config["enabled"]:
                continue
            if not last_contact["from_id"] or not last_contact["context_token"]:
                continue

            now = datetime.now(timezone.utc)
            next_send = None
            if snapshot["progress"]["nextSendAt"]:
                try:
                    next_send = datetime.fromisoformat(
                        snapshot["progress"]["nextSendAt"].replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except ValueError:
                    pass
            if next_send is None:
                clawbot_store.set_proactive_schedule(
                    character_id, next_proactive_time(config, now)
                )
                continue
            if next_send > now:
                continue
            if not proactive_allowed_at(now, config):
                clawbot_store.set_proactive_schedule(
                    character_id, defer_to_proactive_window(now, config)
                )
                continue

            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(
                executor,
                partial(
                    generate_proactive_message,
                    last_contact["from_id"],
                    character_id,
                ),
            )
            reply = str(reply or "").strip()
            if not reply:
                raise ValueError("主动消息模型返回空内容")
            if clawbot_store.active_character()["id"] != character_id:
                continue
            parts = split_reply_messages(reply)
            await send_reply_parts(
                session,
                last_contact["from_id"],
                last_contact["context_token"],
                parts,
                bot_token_ref,
                bot_base_url_ref,
            )
            for part in parts:
                clawbot_store.add_message(
                    last_contact["from_id"],
                    "assistant",
                    part,
                    count_for_memory=False,
                    character_id=character_id,
                )
            clawbot_store.mark_proactive_sent(
                character_id, next_proactive_time(config, now), now
            )
            print(f"[主动消息] 已发送：{reply[:50]}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"[主动消息] 失败：{error}")
            try:
                character_id = clawbot_store.active_character()["id"]
                clawbot_store.mark_proactive_error(
                    character_id,
                    str(error),
                    datetime.now(timezone.utc) + timedelta(minutes=5),
                )
            except Exception as store_error:
                print(f"[主动消息] 无法记录失败状态：{store_error}")


async def do_reconnect(session, bot_token_ref, bot_base_url_ref, last_contact,
                       typing_ticket_cache, reconnect_asked, warning_active,
                       reconnect_in_progress, login_time_ref, cfg):
    """执行重连流程。防重入，失败时优雅降级，成功后原子替换 token。"""
    if reconnect_in_progress[0]:
        return
    reconnect_in_progress[0] = True
    warning_active[0] = False
    reconnect_asked.clear()

    print("[重连] 开始重连流程...")
    from_id = last_contact["from_id"]
    ctx = last_contact["context_token"]

    _base = bot_base_url_ref[0] or BASE_URL
    try:
        data = await fetch_login_qrcode(session, _base, [bot_token_ref[0]] if bot_token_ref[0] else [])
        qrcode = data["qrcode"]
        qrcode_url = data.get("qrcode_img_content", qrcode)
    except Exception as e:
        print(f"[重连] 获取二维码失败: {e}")
        reconnect_in_progress[0] = False
        login_time_ref[0] = time.time()
        return

    # 发送二维码给用户（失败时控制台打印）
    qr_msg = f"[重连] 请扫码完成新连接：{qrcode_url}"
    print(qr_msg)
    render_terminal_qr(qrcode_url)
    await send_msg_safe(session, from_id, ctx, qr_msg, bot_token_ref, bot_base_url_ref)

    # 轮询扫码状态（带超时）
    login_result = await wait_login_confirmation(
        session,
        qrcode,
        _base,
        timeout_seconds=cfg["qrcode_scan_timeout"],
        allow_already_connected=True,
    )
    if login_result.get("already_connected"):
        print("[重连] 服务端提示已连接过此 OpenClaw，继续沿用当前连接")
        new_token = bot_token_ref[0]
        new_base_url = bot_base_url_ref[0]
    else:
        new_token = login_result.get("bot_token")
        new_base_url = login_result.get("baseurl", bot_base_url_ref[0])

    if new_token is None:
        # 扫码超时：重置计时，不 crash
        print("[重连] 扫码超时，重连未完成")
        await send_msg_safe(session, from_id, ctx,
                            "[失败] 扫码超时，重连未完成，下次到期前会再次提醒",
                            bot_token_ref, bot_base_url_ref)
        login_time_ref[0] = time.time()
        reconnect_in_progress[0] = False
        return

    # 成功：原子替换 token 和 base_url
    bot_token_ref[0] = new_token
    bot_base_url_ref[0] = new_base_url
    typing_ticket_cache.clear()
    print("[重连] 新连接已建立，token 已切换")
    await send_msg_safe(session, from_id, ctx,
                        "[完成] 新连接已建立，已自动切换，继续使用",
                        bot_token_ref, bot_base_url_ref)

    reconnect_in_progress[0] = False
    login_time_ref[0] = time.time()


async def reconnect_timer_task(session, bot_token_ref, bot_base_url_ref, last_contact,
                                typing_ticket_cache, reconnect_asked, warning_active,
                                reconnect_in_progress, login_time_ref, cfg):
    """独立定时器任务，与主消息循环并发运行。"""
    while True:
        # 等待到发警告的时间点
        elapsed = time.time() - login_time_ref[0]
        first_wait = max(0, cfg["session_duration"] - cfg["warning_before"] - elapsed)
        await asyncio.sleep(first_wait)

        # 检查剩余时间（可能因测试值设置而已超过 force_before）
        remaining = login_time_ref[0] + cfg["session_duration"] - time.time()
        if remaining <= cfg["force_before"]:
            force_msg = "[自动] 连接即将到期，开始强制重新连接..."
            print(force_msg)
            if not last_contact["from_id"] or not last_contact["context_token"]:
                print("[自动] 尚无最近联系人，跳过本轮自动重连提醒")
                login_time_ref[0] = time.time()
                continue
            await send_msg_safe(session, last_contact["from_id"], last_contact["context_token"],
                                force_msg, bot_token_ref, bot_base_url_ref)
            await do_reconnect(session, bot_token_ref, bot_base_url_ref, last_contact,
                               typing_ticket_cache, reconnect_asked, warning_active,
                               reconnect_in_progress, login_time_ref, cfg)
            continue

        # 发初次警告
        remaining_h = remaining / 3600
        warn_msg = f"[提醒] 连接还剩约 {remaining_h:.1f} 小时到期，是否现在重新连接？回复 Y 立即重连，N 稍后提醒"
        print(warn_msg)
        if not last_contact["from_id"] or not last_contact["context_token"]:
            print("[提醒] 尚无最近联系人，跳过本轮连接到期提醒")
            login_time_ref[0] = time.time()
            continue
        await send_msg_safe(session, last_contact["from_id"], last_contact["context_token"],
                            warn_msg, bot_token_ref, bot_base_url_ref)
        warning_active[0] = True

        # 询问循环
        while True:
            remaining = login_time_ref[0] + cfg["session_duration"] - time.time()
            if remaining <= cfg["force_before"]:
                force_msg = "[自动] 连接即将到期，开始强制重新连接..."
                print(force_msg)
                await send_msg_safe(session, last_contact["from_id"], last_contact["context_token"],
                                    force_msg, bot_token_ref, bot_base_url_ref)
                await do_reconnect(session, bot_token_ref, bot_base_url_ref, last_contact,
                                   typing_ticket_cache, reconnect_asked, warning_active,
                                   reconnect_in_progress, login_time_ref, cfg)
                break

            wait_secs = max(0.0, min(float(cfg["reminder_interval"]),
                                     remaining - cfg["force_before"]))
            try:
                await asyncio.wait_for(reconnect_asked.wait(), timeout=wait_secs)
                # 用户回 Y，执行重连
                await do_reconnect(session, bot_token_ref, bot_base_url_ref, last_contact,
                                   typing_ticket_cache, reconnect_asked, warning_active,
                                   reconnect_in_progress, login_time_ref, cfg)
                break
            except asyncio.TimeoutError:
                # 定时到，重新评估
                remaining = login_time_ref[0] + cfg["session_duration"] - time.time()
                if remaining <= cfg["force_before"]:
                    continue  # 下一轮循环走强制重连分支
                remaining_m = remaining / 60
                remind_msg = (f"[提醒] 连接还剩约 {remaining_m:.0f} 分钟，"
                              f"是否现在重新连接？回复 Y 立即重连，N 继续等待")
                print(remind_msg)
                # 用最新的 last_contact（可能已更新）
                await send_msg_safe(session, last_contact["from_id"], last_contact["context_token"],
                                    remind_msg, bot_token_ref, bot_base_url_ref)


def render_terminal_qr(content: str):
    if not content:
        return
    print("\n扫码地址:", content)
    if content.startswith("http") and render_terminal_image_from_url(content):
        return
    render_generated_qr(content)


def render_terminal_image_from_url(url: str) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        image = Image.open(io.BytesIO(data)).convert("L")
        max_width = 72
        scale = max(1, int(image.width / max_width))
        width = max(1, int(image.width / scale))
        height = max(1, int(image.height / scale))
        image = image.resize((width, height))
        print()
        for y in range(height):
            print("".join("██" if image.getpixel((x, y)) < 128 else "  " for x in range(width)))
        print()
        return True
    except Exception as e:
        print(f"二维码图片渲染失败，改用本地二维码生成方式: {e}")
        return False


def render_generated_qr(content: str):
    try:
        import qrcode
    except ImportError:
        print("未安装 qrcode/Pillow，无法在终端渲染二维码；安装 `pip install qrcode pillow` 后会自动显示。")
        return

    qr = qrcode.QRCode(border=1)
    qr.add_data(content)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    print()
    for row in matrix:
        print("".join("██" if cell else "  " for cell in row))
    print()


def save_qrcode_content(content: str):
    if not content:
        return

    def save_generated_png(raw_content: str) -> bool:
        try:
            import qrcode
            img = qrcode.make(raw_content)
            img.save("qrcode.png")
            print("qrcode.png saved. Open it and scan with WeChat.")
            return True
        except Exception as e:
            print(f"Failed to generate qrcode.png: {e}")
            return False

    if content.startswith("data:image/"):
        header, b64 = content.split(",", 1)
        m = re.search(r"data:image/(\w+)", header)
        ext = m.group(1) if m else "png"
        with open(f"qrcode.{ext}", "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"二维码已保存到 qrcode.{ext}")
    elif content.startswith("<svg"):
        with open("qrcode.svg", "w", encoding="utf-8") as f:
            f.write(content)
        print("二维码已保存到 qrcode.svg，用浏览器打开")
    elif content.startswith("http"):
        if not save_generated_png(content):
            render_terminal_qr(content)
    else:
        try:
            with open("qrcode.png", "wb") as f:
                f.write(base64.b64decode(content))
            print("二维码已保存到 qrcode.png")
        except Exception:
            if not save_generated_png(content):
                render_terminal_qr(content)


async def fetch_login_qrcode(session, base_url=BASE_URL, local_token_list=None):
    body = {"local_token_list": local_token_list or []}
    data = await api_post(session, "ilink/bot/get_bot_qrcode?bot_type=3", body, None, base_url)
    if data.get("qrcode"):
        return data
    print("POST 获取二维码未返回 qrcode，尝试兼容旧版 GET 流程。")
    return await api_get(session, "ilink/bot/get_bot_qrcode?bot_type=3", None, base_url)


async def poll_login_status(session, qrcode, base_url=BASE_URL, verify_code=None):
    endpoint = f"ilink/bot/get_qrcode_status?qrcode={quote(qrcode, safe='')}"
    if verify_code:
        endpoint += f"&verify_code={quote(verify_code, safe='')}"
    status = await api_get(session, endpoint, None, base_url)
    state = status.get("status", "")

    if state == "confirmed" or status.get("bot_token"):
        return {
            "bot_token": status.get("bot_token"),
            "baseurl": status.get("baseurl") or status.get("base_url") or base_url,
            "ilink_bot_id": status.get("ilink_bot_id"),
            "ilink_user_id": status.get("ilink_user_id"),
        }
    if state == "binded_redirect" or status.get("binded_redirect"):
        return {"already_connected": True}
    if state == "expired":
        return {"expired": True}
    if state == "scaned_but_redirect":
        redirect_host = status.get("redirect_host")
        if redirect_host:
            return {"redirect_base": f"https://{redirect_host}"}
        print("服务端要求切换扫码轮询节点，但未返回 redirect_host，继续使用当前节点。")
        return {}
    if state == "scaned":
        return {"scanned": True, "verify_code_accepted": bool(verify_code)}
    elif state in ("need_verifycode", "verify_code_blocked") or status.get("need_verifycode"):
        if state == "verify_code_blocked":
            return {"verify_code_blocked": True}
        return {"need_verifycode": True, "retry_verifycode": bool(verify_code)}
    elif state and state != "wait":
        print(f"登录状态: {state}，原始响应: {status}")

    if status.get("local_token_list"):
        print("服务端返回 local_token_list 信息，继续等待扫码确认。")
    return {}


async def wait_login_confirmation(session, qrcode, base_url=BASE_URL, timeout_seconds=None,
                                  allow_already_connected=False):
    deadline = time.time() + timeout_seconds if timeout_seconds else None
    current_base_url = base_url
    pending_verify_code = None
    scanned_printed = False

    while True:
        if deadline and time.time() >= deadline:
            return {"timeout": True}

        try:
            result = await poll_login_status(session, qrcode, current_base_url, pending_verify_code)
        except Exception as e:
            print(f"轮询扫码状态失败，稍后重试: {e}")
            await asyncio.sleep(1)
            continue

        if result.get("bot_token"):
            return result
        if result.get("already_connected"):
            return result if allow_already_connected else {"already_connected": True}
        if result.get("expired"):
            return result
        if result.get("verify_code_blocked"):
            return result
        if result.get("redirect_base"):
            current_base_url = result["redirect_base"]
            print(f"扫码轮询切换到新节点: {current_base_url}")
            continue
        if result.get("scanned"):
            if pending_verify_code and result.get("verify_code_accepted"):
                pending_verify_code = None
            if not scanned_printed:
                print("已扫码，等待手机端确认...")
                scanned_printed = True
        if result.get("need_verifycode"):
            prompt = "你输入的数字不匹配，请重新输入: " if result.get("retry_verifycode") else "请输入手机微信显示的数字配对码: "
            pending_verify_code = input(prompt).strip()
            continue

        await asyncio.sleep(1)


async def login_with_qrcode(session, base_url=BASE_URL):
    refresh_count = 0
    max_refresh_count = 3
    while True:
        data = await fetch_login_qrcode(session, base_url)
        qrcode = data["qrcode"]
        qrcode_img_content = data.get("qrcode_img_content", "")

        print("qrcode:", qrcode)
        save_qrcode_content(str(qrcode_img_content or qrcode))
        print("等待扫码...")

        login_result = await wait_login_confirmation(session, qrcode, base_url)
        if login_result.get("bot_token"):
            return login_result
        if login_result.get("already_connected"):
            print("服务端提示此端已连接过，但当前独立程序没有可复用 token，将重新生成二维码。")
        elif login_result.get("expired"):
            print("二维码已过期，正在重新生成...")
        elif login_result.get("verify_code_blocked"):
            print("多次输入配对码错误，正在刷新二维码...")
        elif login_result.get("timeout"):
            print("登录等待超时，正在重新生成二维码...")

        refresh_count += 1
        if refresh_count >= max_refresh_count:
            raise RuntimeError("二维码多次失效或登录失败，请稍后重试。")


async def process_chat_batch(
    session,
    batch,
    typing_ticket_cache,
    bot_token_ref,
    bot_base_url_ref,
):
    from_id = batch["from_id"]
    context_token = batch["context_token"]
    character_id = batch["character_id"]
    model_key = batch["model_key"]
    text = "\n".join(batch["messages"])
    typing_ticket = ""
    try:
        if from_id not in typing_ticket_cache:
            cfg = await api_post(
                session,
                "ilink/bot/getconfig",
                {
                    "ilink_user_id": from_id,
                    "context_token": context_token,
                    "base_info": base_info(),
                },
                bot_token_ref[0],
                bot_base_url_ref[0] or None,
            )
            typing_ticket_cache[from_id] = cfg.get("typing_ticket", "")
        typing_ticket = typing_ticket_cache[from_id]
        if typing_ticket:
            await api_post(
                session,
                "ilink/bot/sendtyping",
                {
                    "ilink_user_id": from_id,
                    "typing_ticket": typing_ticket,
                    "status": 1,
                    "base_info": base_info(),
                },
                bot_token_ref[0],
                bot_base_url_ref[0] or None,
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            partial(
                generate_chat_result,
                text,
                character_id,
                model_key,
                from_id,
            ),
        )
        parts = split_reply_messages(result["reply"])
        if not parts:
            raise ValueError("AI returned an empty reply")
        await send_reply_parts(
            session,
            from_id,
            context_token,
            parts,
            bot_token_ref,
            bot_base_url_ref,
        )

        clawbot_store.add_message(
            from_id, "user", text, character_id=character_id
        )
        for part in parts:
            clawbot_store.add_message(
                from_id, "assistant", part, character_id=character_id
            )
        try:
            clawbot_store.apply_affection_delta(
                character_id,
                result["affectionDelta"],
                result["affectionReason"],
            )
        except Exception as error:
            print(f"Failed to update affection: {error}")
        memory_failed = await process_memory_tasks(
            loop, character_id, from_id
        )
        if memory_failed:
            await send_msg_safe(
                session,
                from_id,
                context_token,
                "记忆总结失败，稍后自动重试。",
                bot_token_ref,
                bot_base_url_ref,
            )
        print(f"已完成一轮回复，共 {len(parts)} 条消息")
    except Exception as error:
        print(f"处理聊天批次失败: {error}")
    finally:
        if typing_ticket:
            try:
                await api_post(
                    session,
                    "ilink/bot/sendtyping",
                    {
                        "ilink_user_id": from_id,
                        "typing_ticket": typing_ticket,
                        "status": 2,
                        "base_info": base_info(),
                    },
                    bot_token_ref[0],
                    bot_base_url_ref[0] or None,
                )
            except Exception as error:
                print(f"取消输入状态失败: {error}")


def enqueue_chat_message(
    buffers,
    session,
    from_id,
    context_token,
    text,
    typing_ticket_cache,
    bot_token_ref,
    bot_base_url_ref,
):
    state = buffers.setdefault(
        from_id,
        {"messages": [], "timer": None},
    )
    if not state["messages"]:
        character_id = clawbot_store.active_character()["id"]
        state["character_id"] = character_id
        state["model_key"] = get_character_model_key(character_id)
    state["messages"].append(text)
    state["context_token"] = context_token
    if state["timer"] and not state["timer"].done():
        state["timer"].cancel()
    wait_seconds = clawbot_store.general_snapshot(
        state["character_id"]
    )["config"]["mergeWaitSeconds"]

    async def flush_after_wait():
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            return
        batch = {
            "from_id": from_id,
            "context_token": state["context_token"],
            "character_id": state["character_id"],
            "model_key": state["model_key"],
            "messages": list(state["messages"]),
        }
        state["messages"].clear()
        state["timer"] = None
        asyncio.create_task(
            process_chat_batch(
                session,
                batch,
                typing_ticket_cache,
                bot_token_ref,
                bot_base_url_ref,
            )
        )

    state["timer"] = asyncio.create_task(flush_after_wait())


async def main():
    try:
        await start_backend(clawbot_store, model_loader=get_model_profiles)
    except OSError as error:
        if getattr(error, "winerror", None) == 10048:
            print("ClawBot backend port is already in use; continuing with the existing backend.")
        else:
            raise
    async with aiohttp.ClientSession() as session:
        # 1. 获取二维码并等待扫码
        login_result = await login_with_qrcode(session)
        bot_token = login_result["bot_token"]
        bot_base_url = login_result.get("baseurl", "")
        print(f"登录成功！baseurl={bot_base_url}")
        # 3. 消息循环所需状态
        bot_token_ref = [bot_token]
        bot_base_url_ref = [bot_base_url]
        typing_ticket_cache = {}
        chat_buffers = {}
        last_contact = {"from_id": "", "context_token": ""}
        asyncio.create_task(
            proactive_message_task(
                session, bot_token_ref, bot_base_url_ref, last_contact
            )
        )

        # 4. 长轮询收消息
        get_updates_buf = ""
        print("开始监听消息...")
        while True:
            result = await api_post(
                session,
                "ilink/bot/getupdates",
                {"get_updates_buf": get_updates_buf, "base_info": base_info()},
                bot_token_ref[0],
                bot_base_url_ref[0] or None,
            )
            get_updates_buf = result.get("get_updates_buf") or get_updates_buf

            for msg in result.get("msgs") or []:
                if msg.get("message_type") != 1:
                    continue
                text = msg.get("item_list", [{}])[0].get("text_item", {}).get("text", "")
                from_id = msg["from_user_id"]
                context_token = msg["context_token"]
                last_contact["from_id"] = from_id
                last_contact["context_token"] = context_token
                print(f"收到消息: {text}")

                is_model_command, model_reply = handle_model_command(text)
                if is_model_command:
                    await send_msg_safe(session, from_id, context_token,
                                        model_reply or format_model_list(),
                                        bot_token_ref, bot_base_url_ref)
                    continue

                is_memory_command, memory_reply = handle_memory_command(text)
                if is_memory_command:
                    await send_msg_safe(
                        session,
                        from_id,
                        context_token,
                        memory_reply or format_memory_settings(),
                        bot_token_ref,
                        bot_base_url_ref,
                    )
                    continue

                enqueue_chat_message(
                    chat_buffers,
                    session,
                    from_id,
                    context_token,
                    text,
                    typing_ticket_cache,
                    bot_token_ref,
                    bot_base_url_ref,
                )


if __name__ == "__main__":
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║          微信 ClawBot  ·  WeChat iLink Bot               ║\n"
        "║  Copyright (c) 2026 SiverKing. All rights reserved.     ║\n"
        "║  GitHub : https://github.com/SiverKing/weixin-ClawBot-API║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )
    _raw_cfg = load_or_create_config()
    ai = create_ai_client(_raw_cfg)
    asyncio.run(main())
