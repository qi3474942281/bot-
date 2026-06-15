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


class ChatBehaviorTests(unittest.TestCase):
    def test_short_reply_stays_in_one_message(self):
        self.assertEqual(bot.split_reply_messages("好，我知道了。"), ["好，我知道了。"])

    def test_medium_reply_prefers_two_messages(self):
        text = (
            "我刚刚认真看完你说的内容，确实能理解你为什么会觉得有点累。"
            "先别急着逼自己马上做决定，我们可以把最难的部分慢慢拆开。"
            "你愿意的话就先告诉我，现在最让你难受的是哪一件事？"
        )
        parts = bot.split_reply_messages(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual("".join(parts), text)

    def test_long_reply_preserves_all_content(self):
        text = "".join(f"这是第{number}句话。" for number in range(30))
        parts = bot.split_reply_messages(text)
        self.assertGreater(len(parts), 2)
        self.assertEqual("".join(parts), text)

    def test_character_model_is_stored_per_character(self):
        bot.clawbot_store = clawbot_backend.ClawBotStore(":memory:")
        models = {
            "fast": {"model": "fast-model"},
            "smart": {"model": "smart-model"},
        }
        with (
            patch.object(bot, "get_model_profiles", return_value=models),
            patch.object(bot, "get_active_model_key", return_value="fast"),
        ):
            bot.save_character_model_key("default", "smart")
            self.assertEqual(
                bot.get_character_model_key("default"), "smart"
            )


if __name__ == "__main__":
    unittest.main()
