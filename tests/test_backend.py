import unittest
from datetime import datetime, timezone

from aiohttp.test_utils import TestClient, TestServer

from clawbot_backend import (
    ClawBotStore,
    create_api_app,
    defer_to_proactive_window,
    proactive_allowed_at,
)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = ClawBotStore(":memory:")
        app = create_api_app(
            self.store,
            {
                "token": "test-token",
                "allowed_origins": ["https://qi3474942281.github.io"],
            },
            model_loader=lambda: {
                "fast": {"model": "fast-model"},
                "smart": {"model": "smart-model"},
            },
        )
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.headers = {
            "Authorization": "Bearer test-token",
            "Origin": "https://qi3474942281.github.io",
        }

    async def asyncTearDown(self):
        await self.client.close()

    async def test_rejects_missing_token(self):
        response = await self.client.get("/api/state")
        self.assertEqual(response.status, 401)

    async def test_saves_state_and_builds_prompt(self):
        state = {
            "activeCharacterId": "main",
            "characters": [
                {
                    "id": "main",
                    "name": "测试角色",
                    "persona": {
                        "description": "说话简洁。",
                        "traits": "诚实",
                        "examples": "用户：你好\n助手：你好。",
                    },
                    "memories": ["用户喜欢咖啡"],
                    "promptMemories": ["不要编造事实"],
                }
            ],
        }
        response = await self.client.put(
            "/api/state",
            headers=self.headers,
            json={"state": state},
        )
        self.assertEqual(response.status, 200)
        prompt = self.store.build_prompt("基础提示词")
        self.assertIn("测试角色", prompt)
        self.assertIn("用户喜欢咖啡", prompt)
        self.assertIn("不要编造事实", prompt)

    async def test_records_and_returns_history(self):
        self.store.add_message("user-1", "user", "你好")
        self.store.add_message("user-1", "assistant", "你好。")
        response = await self.client.get("/api/state", headers=self.headers)
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(len(payload["history"]), 2)
        recent = self.store.recent_history("user-1")
        self.assertEqual([item["role"] for item in recent], ["user", "assistant"])

    async def test_rejects_invalid_state(self):
        response = await self.client.put(
            "/api/state",
            headers=self.headers,
            json={"state": {"characters": []}},
        )
        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertIn("character", payload["error"])

    async def test_migrates_legacy_memories_and_injects_all_memory_types(self):
        state = {
            "activeCharacterId": "main",
            "characters": [
                {
                    "id": "main",
                    "name": "记忆角色",
                    "persona": {},
                    "memories": ["用户手动事实"],
                    "promptMemories": ["必须守约"],
                }
            ],
        }
        self.store.save_state(state)
        snapshot = self.store.memory_snapshot("main")
        self.assertEqual(snapshot["facts"][0]["content"], "用户手动事实")
        self.assertEqual(snapshot["facts"][0]["source"], "user")
        self.store.add_memory_entry(
            "main",
            {
                "kind": "stm",
                "timeLabel": "今天",
                "scene": "咖啡店",
                "content": "用户点了一杯咖啡并讨论旅行计划。",
            },
        )
        self.store.add_memory_entry(
            "main",
            {
                "kind": "ltm",
                "timeLabel": "本月",
                "scene": "日常",
                "content": "用户持续规划旅行并偏好咖啡店场景。",
            },
        )
        prompt = self.store.build_prompt("基础提示词")
        for expected in ("必须守约", "用户手动事实", "咖啡店", "持续规划旅行"):
            self.assertIn(expected, prompt)

    async def test_memory_config_version_and_user_edit_protection(self):
        snapshot = self.store.memory_snapshot("default")
        config = dict(snapshot["config"])
        config["factRounds"] = 12
        updated = self.store.save_memory_config(
            "default", config, snapshot["version"]
        )
        self.assertEqual(updated["config"]["factRounds"], 12)
        with self.assertRaises(ValueError):
            self.store.save_memory_config(
                "default", config, snapshot["version"]
            )

        entry = self.store.add_memory_entry(
            "default",
            {"kind": "fact", "content": "AI 事实", "source": "ai"},
        )
        edited = self.store.update_memory_entry(
            "default", entry["id"], {"content": "用户修正后的事实"}
        )
        self.assertEqual(edited["source"], "user")

    async def test_memory_tasks_ltm_retry_and_next_cycle_clear(self):
        snapshot = self.store.memory_snapshot("default")
        config = dict(snapshot["config"])
        config.update(
            {
                "factRounds": 2,
                "stmRounds": 2,
                "stmMinChars": 10,
                "stmMaxChars": 100,
                "ltmAfterStm": 2,
                "ltmMinChars": 20,
                "ltmMaxChars": 200,
            }
        )
        self.store.save_memory_config("default", config, snapshot["version"])

        def add_round(number):
            self.store.add_message("user-1", "user", f"问题 {number}")
            self.store.add_message("user-1", "assistant", f"回答 {number}")
            return self.store.prepare_memory_tasks("default", "user-1")

        self.assertEqual(add_round(1), [])
        tasks = add_round(2)
        fact_task = next(item for item in tasks if item["kind"] == "fact")
        stm_task = next(item for item in tasks if item["kind"] == "stm")
        self.store.complete_fact_task(
            fact_task, {"action": "add", "content": "用户连续提出两个测试问题。"}
        )
        needs_ltm = self.store.complete_stm_task(
            stm_task,
            {"timeLabel": "第一阶段", "scene": "测试", "content": "用户连续提出问题，助手依次完成回答。"},
        )
        self.assertFalse(needs_ltm)

        self.assertEqual(add_round(3), [])
        tasks = add_round(4)
        fact_task = next(item for item in tasks if item["kind"] == "fact")
        stm_task = next(item for item in tasks if item["kind"] == "stm")
        self.store.complete_fact_task(
            fact_task,
            {
                "action": "merge",
                "entryId": self.store.memory_snapshot("default")["facts"][0]["id"],
                "content": "用户连续提出四个测试问题。",
            },
        )
        needs_ltm = self.store.complete_stm_task(
            stm_task,
            {"timeLabel": "第二阶段", "scene": "测试", "content": "用户继续提出问题，助手保持连续回应。"},
        )
        self.assertTrue(needs_ltm)
        retry_tasks = self.store.prepare_memory_tasks("default", "user-1")
        self.assertIn("ltm", [item["kind"] for item in retry_tasks])
        self.assertNotIn("stm", [item["kind"] for item in retry_tasks])
        retry_tasks = add_round(5)
        self.assertIn("ltm", [item["kind"] for item in retry_tasks])
        self.assertNotIn("stm", [item["kind"] for item in retry_tasks])
        ltm_task = next(item for item in retry_tasks if item["kind"] == "ltm")
        self.store.complete_ltm_task(
            ltm_task,
            {
                "timeLabel": "第一至第二阶段",
                "scene": "连续测试",
                "content": "用户在两个阶段连续提出四个测试问题，助手按照先后顺序逐一回应并保持上下文连续。",
            },
        )
        self.assertTrue(
            self.store.memory_snapshot("default")["progress"]["clearStmBeforeNext"]
        )
        tasks = add_round(6)
        self.assertEqual(
            len(self.store.memory_snapshot("default")["stm"]), 0
        )
        self.assertIn("stm", [item["kind"] for item in tasks])

    async def test_memory_api_crud_and_version_conflict(self):
        response = await self.client.get(
            "/api/memory/default", headers=self.headers
        )
        self.assertEqual(response.status, 200)
        snapshot = await response.json()
        response = await self.client.post(
            "/api/memory/default/entries",
            headers=self.headers,
            json={
                "kind": "fact",
                "content": "API 添加的事实",
                "version": snapshot["version"],
            },
        )
        self.assertEqual(response.status, 201)
        entry = await response.json()
        current = self.store.memory_snapshot("default")
        response = await self.client.put(
            f"/api/memory/default/entries/{entry['id']}",
            headers=self.headers,
            json={
                "content": "API 更新的事实",
                "version": current["version"],
            },
        )
        self.assertEqual(response.status, 200)
        response = await self.client.put(
            "/api/memory/default/config",
            headers=self.headers,
            json={"config": snapshot["config"], "version": snapshot["version"]},
        )
        self.assertEqual(response.status, 409)
        response = await self.client.delete(
            f"/api/memory/default/entries/{entry['id']}",
            headers=self.headers,
            json={
                "version": self.store.memory_snapshot("default")["version"]
            },
        )
        self.assertEqual(response.status, 200)

    async def test_new_profile_starts_after_existing_history(self):
        with self.store.lock, self.store._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(character_id, user_id, role, content, created_at)
                VALUES('legacy', 'user-1', 'user', '旧问题', '2026-01-01T00:00:00+00:00')
                """
            )
            connection.execute(
                """
                INSERT INTO messages(character_id, user_id, role, content, created_at)
                VALUES('legacy', 'user-1', 'assistant', '旧回答', '2026-01-01T00:00:01+00:00')
                """
            )
        self.store.save_state(
            {
                "activeCharacterId": "legacy",
                "characters": [
                    {
                        "id": "legacy",
                        "name": "旧角色",
                        "persona": {},
                        "memories": [],
                        "promptMemories": [],
                    }
                ],
            }
        )
        snapshot = self.store.memory_snapshot("legacy")
        self.assertEqual(snapshot["progress"]["factCheckpoint"], 2)
        self.assertEqual(snapshot["progress"]["stmCheckpoint"], 2)
        self.store.add_message("user-1", "user", "新问题")
        self.store.add_message("user-1", "assistant", "新回答")
        config = dict(snapshot["config"])
        config.update({"factRounds": 1, "stmRounds": 1})
        snapshot = self.store.save_memory_config(
            "legacy", config, snapshot["version"]
        )
        tasks = self.store.prepare_memory_tasks("legacy", "user-1")
        for task in tasks:
            if "messages" in task:
                self.assertEqual(
                    [item["content"] for item in task["messages"]],
                    ["新问题", "新回答"],
                )

    async def test_proactive_config_api_and_version_conflict(self):
        response = await self.client.get(
            "/api/proactive/default", headers=self.headers
        )
        self.assertEqual(response.status, 200)
        snapshot = await response.json()
        config = dict(snapshot["config"])
        config.update(
            {
                "enabled": True,
                "mode": "exact",
                "exactMinutes": 15,
                "windowStart": "09:00",
                "windowEnd": "22:30",
            }
        )
        response = await self.client.put(
            "/api/proactive/default",
            headers=self.headers,
            json={"config": config, "version": snapshot["version"]},
        )
        self.assertEqual(response.status, 200)
        updated = await response.json()
        self.assertTrue(updated["config"]["enabled"])
        self.assertEqual(updated["config"]["exactMinutes"], 15)
        self.assertEqual(updated["progress"]["nextSendAt"], "")

        response = await self.client.put(
            "/api/proactive/default",
            headers=self.headers,
            json={"config": config, "version": snapshot["version"]},
        )
        self.assertEqual(response.status, 409)

    async def test_proactive_schedule_and_non_counting_message(self):
        snapshot = self.store.proactive_snapshot("default")
        config = dict(snapshot["config"])
        config["enabled"] = True
        self.store.save_proactive_config(
            "default", config, snapshot["version"]
        )
        due = datetime(2026, 6, 14, 4, 0, tzinfo=timezone.utc)
        self.store.set_proactive_schedule("default", due)
        self.assertTrue(
            self.store.proactive_due(
                "default", datetime(2026, 6, 14, 4, 1, tzinfo=timezone.utc)
            )
        )
        self.store.add_message(
            "user-1", "assistant", "主动问候", count_for_memory=False
        )
        self.assertEqual(self.store.prepare_memory_tasks("default", "user-1"), [])
        self.assertEqual(
            self.store.memory_snapshot("default")["progress"]["factRounds"], 0
        )

    async def test_proactive_china_time_window_including_overnight(self):
        daytime = {"windowStart": "08:00", "windowEnd": "23:00"}
        before = datetime(2026, 6, 13, 23, 30, tzinfo=timezone.utc)
        deferred = defer_to_proactive_window(before, daytime)
        self.assertEqual(
            deferred, datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)
        )
        self.assertFalse(proactive_allowed_at(before, daytime))
        self.assertTrue(
            proactive_allowed_at(
                datetime(2026, 6, 14, 2, 0, tzinfo=timezone.utc), daytime
            )
        )

        overnight = {"windowStart": "22:00", "windowEnd": "06:00"}
        self.assertTrue(
            proactive_allowed_at(
                datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc), overnight
            )
        )
        self.assertFalse(
            proactive_allowed_at(
                datetime(2026, 6, 14, 4, 0, tzinfo=timezone.utc), overnight
            )
        )

    async def test_general_config_is_per_character_and_versioned(self):
        models = {
            "fast": {"model": "fast-model"},
            "smart": {"model": "smart-model"},
        }
        snapshot = self.store.general_snapshot("default")
        config = dict(snapshot["config"])
        config.update({"mergeWaitSeconds": 9, "currentModel": "smart"})
        updated = self.store.save_general_config(
            "default", config, snapshot["version"], models
        )
        self.assertEqual(updated["config"]["mergeWaitSeconds"], 9)
        self.assertEqual(updated["config"]["currentModel"], "smart")
        with self.assertRaises(ValueError):
            self.store.save_general_config(
                "default", config, snapshot["version"], models
            )
        with self.assertRaises(ValueError):
            self.store.save_general_config(
                "default",
                {**config, "currentModel": "missing"},
                updated["version"],
                models,
            )

        response = await self.client.get(
            "/api/general/default", headers=self.headers
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertIn("smart", payload["models"])
        response = await self.client.put(
            "/api/general/default",
            headers=self.headers,
            json={
                "config": {
                    "mergeWaitSeconds": 7,
                    "currentModel": "fast",
                },
                "version": payload["version"],
            },
        )
        self.assertEqual(response.status, 200)

    async def test_affection_manual_save_delta_limits_and_recent_history(self):
        snapshot = self.store.affection_snapshot("default")
        self.assertEqual(snapshot["value"], 0)
        saved = self.store.save_affection(
            "default", 490, snapshot["version"]
        )
        self.assertEqual(saved["value"], 490)
        self.assertEqual(saved["history"], [])

        self.store.apply_affection_delta("default", 20, "明确规则奖励")
        self.assertEqual(self.store.affection_snapshot("default")["value"], 500)
        self.store.apply_affection_delta("default", -600, "严重负面互动")
        self.assertEqual(self.store.affection_snapshot("default")["value"], 0)
        for number in range(12):
            self.store.apply_affection_delta(
                "default", 1, f"变化 {number}"
            )
        recent = self.store.affection_snapshot("default")["history"]
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[0]["reason"], "变化 11")

        response = await self.client.get(
            "/api/affection/default", headers=self.headers
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        response = await self.client.put(
            "/api/affection/default",
            headers=self.headers,
            json={"value": 123, "version": payload["version"]},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["value"], 123)

    async def test_multiple_assistant_bubbles_count_as_one_memory_round(self):
        snapshot = self.store.memory_snapshot("default")
        config = dict(snapshot["config"])
        config.update({"factRounds": 2, "stmRounds": 2})
        self.store.save_memory_config(
            "default", config, snapshot["version"]
        )
        self.store.add_message("user-1", "user", "合并后的用户消息")
        self.store.add_message("user-1", "assistant", "第一条回复")
        self.store.add_message("user-1", "assistant", "第二条回复")
        self.assertEqual(
            self.store.prepare_memory_tasks("default", "user-1"), []
        )
        progress = self.store.memory_snapshot("default")["progress"]
        self.assertEqual(progress["factRounds"], 1)
        self.assertEqual(progress["stmRounds"], 1)


if __name__ == "__main__":
    unittest.main()
