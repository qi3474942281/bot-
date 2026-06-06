import unittest

from aiohttp.test_utils import TestClient, TestServer

from clawbot_backend import ClawBotStore, create_api_app


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = ClawBotStore(":memory:")
        app = create_api_app(
            self.store,
            {
                "token": "test-token",
                "allowed_origins": ["https://qi3474942281.github.io"],
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


if __name__ == "__main__":
    unittest.main()
