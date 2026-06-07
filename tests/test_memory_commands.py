import importlib
import sys
import unittest
from unittest.mock import patch

import clawbot_backend
import sync_from_github


TEST_STORE = clawbot_backend.ClawBotStore(":memory:")

with (
    patch.object(sync_from_github, "sync_public_settings", return_value=False),
    patch.object(clawbot_backend, "ClawBotStore", return_value=TEST_STORE),
):
    sys.modules.pop("bot", None)
    bot = importlib.import_module("bot")


class MemoryCommandTests(unittest.TestCase):
    def setUp(self):
        bot.clawbot_store = clawbot_backend.ClawBotStore(":memory:")
        self.models_patch = patch.object(
            bot,
            "get_model_profiles",
            return_value={
                "fast": {"model": "fast-model"},
                "smart": {"model": "smart-model"},
            },
        )
        self.models_patch.start()

    def tearDown(self):
        self.models_patch.stop()

    def test_memory_help_and_numeric_setting(self):
        handled, reply = bot.handle_memory_command("/memory")
        self.assertTrue(handled)
        self.assertIn("fact-rounds", reply)

        handled, reply = bot.handle_memory_command(
            "/memory set fact-rounds 12"
        )
        self.assertTrue(handled)
        self.assertIn("已更新", reply)
        config = bot.clawbot_store.memory_snapshot("default")["config"]
        self.assertEqual(config["factRounds"], 12)

    def test_memory_model_and_switch_settings(self):
        handled, reply = bot.handle_memory_command(
            "/memory set model smart"
        )
        self.assertTrue(handled)
        self.assertIn("已更新", reply)
        bot.handle_memory_command("/memory set facts-auto off")
        config = bot.clawbot_store.memory_snapshot("default")["config"]
        self.assertEqual(config["summaryModel"], "smart")
        self.assertFalse(config["factsAuto"])

    def test_invalid_memory_command_does_not_change_config(self):
        before = bot.clawbot_store.memory_snapshot("default")["config"]
        handled, reply = bot.handle_memory_command(
            "/memory set model missing"
        )
        after = bot.clawbot_store.memory_snapshot("default")["config"]
        self.assertTrue(handled)
        self.assertIn("未修改", reply)
        self.assertEqual(before, after)

    def test_non_memory_message_is_not_a_command(self):
        self.assertEqual(
            bot.handle_memory_command("把短期记忆改成 20 轮"),
            (False, None),
        )

    def test_summary_validation_rejects_bad_length_and_manual_merge(self):
        stm_task = {
            "kind": "stm",
            "config": {"stmMinChars": 10, "stmMaxChars": 20},
        }
        with self.assertRaises(ValueError):
            bot._validate_summary_result(
                stm_task,
                {"timeLabel": "今天", "scene": "测试", "content": "太短"},
            )

        fact_task = {
            "kind": "fact",
            "existingAiFacts": [{"id": 3, "content": "AI 事实"}],
        }
        with self.assertRaises(ValueError):
            bot._validate_summary_result(
                fact_task,
                {"action": "merge", "entryId": 9, "content": "不能改人工事实"},
            )


if __name__ == "__main__":
    unittest.main()
