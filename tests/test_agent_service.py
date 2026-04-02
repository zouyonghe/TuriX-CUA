import unittest
import base64
import io
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agent.service import (
    Agent,
    build_google_images_fallback_actions,
    llm_supports_response_format,
    rewrite_google_images_navigation_actions,
    screenshot_to_dataurl,
    strip_premature_done_actions,
    to_structured,
)
from src.agent.views import AgentOutput
from src.controller.registry.views import ActionModel
from src.utils.llm_response import normalize_llm_json_text


class DummyOutput(BaseModel):
    value: str


_EXAMPLES_MAIN_SPEC = spec_from_file_location(
    "examples_main_for_tests",
    Path(__file__).resolve().parents[1] / "examples" / "main.py",
)
_EXAMPLES_MAIN = module_from_spec(_EXAMPLES_MAIN_SPEC)
assert _EXAMPLES_MAIN_SPEC.loader is not None
_EXAMPLES_MAIN_SPEC.loader.exec_module(_EXAMPLES_MAIN)


class ResponseFormatSupportTests(unittest.TestCase):
    def test_custom_chatopenai_proxy_disables_response_format(self) -> None:
        llm = ChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        self.assertFalse(llm_supports_response_format(llm))

    def test_openai_cloud_keeps_response_format_enabled(self) -> None:
        llm = ChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
        )

        self.assertTrue(llm_supports_response_format(llm))

    def test_to_structured_returns_original_llm_for_custom_proxy(self) -> None:
        llm = ChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        wrapped = to_structured(
            llm,
            {"type": "json_schema", "json_schema": {"name": "dummy", "schema": {}}},
            DummyOutput,
        )

        self.assertIs(wrapped, llm)

    def test_build_openai_compatible_llm_disables_response_format_for_custom_proxy(self) -> None:
        llm = _EXAMPLES_MAIN.build_openai_compatible_llm(
            model_name="gpt-5.4",
            api_key="test-key",
            base_url="https://proxy.pieixan.icu/v1",
        )

        self.assertFalse(getattr(llm, "_turix_supports_response_format"))

    def test_build_openai_compatible_llm_keeps_response_format_for_openai_cloud(self) -> None:
        llm = _EXAMPLES_MAIN.build_openai_compatible_llm(
            model_name="gpt-5.4",
            api_key="test-key",
            base_url=None,
        )

        self.assertTrue(getattr(llm, "_turix_supports_response_format"))

    def test_log_response_without_current_state_does_not_raise(self) -> None:
        agent = Agent.__new__(Agent)
        agent.brain_memory = ""
        agent.next_goal = ""

        Agent._log_response(agent, AgentOutput(action=[]))

    def test_compat_chat_openai_wraps_string_response(self) -> None:
        llm = _EXAMPLES_MAIN.CompatChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        result = llm._create_chat_result('{"action":[{"done":{}}]}')

        self.assertEqual(result.generations[0].message.content, '{"action":[{"done":{}}]}')

    def test_compat_chat_openai_combines_outputs_without_token_usage(self) -> None:
        llm = _EXAMPLES_MAIN.CompatChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        combined = llm._combine_llm_outputs(
            [
                {},
                {"token_usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
                {"system_fingerprint": "fp-test"},
            ]
        )

        self.assertEqual(combined["token_usage"]["prompt_tokens"], 12)
        self.assertEqual(combined["token_usage"]["completion_tokens"], 3)
        self.assertEqual(combined["token_usage"]["total_tokens"], 15)
        self.assertEqual(combined["system_fingerprint"], "fp-test")

    def test_strip_premature_done_actions_keeps_done_only_steps(self) -> None:
        done_only = [ActionModel.model_validate({"done": {}})]

        filtered = strip_premature_done_actions(done_only)

        self.assertEqual([action.model_dump(exclude_unset=True) for action in filtered], [{"done": {}}])

    def test_strip_premature_done_actions_removes_done_when_other_actions_exist(self) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [97, 104]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"done": {}}),
        ]

        filtered = strip_premature_done_actions(actions)

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in filtered],
            [{"Click": {"position": [97, 104]}}, {"wait": {}}],
        )

    def test_normalize_llm_json_text_extracts_json_from_sse_chunks(self) -> None:
        text = (
            'data: {"choices":[{"delta":{"content":"{\\"action\\":"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"[{\\"done\\":{}}]}"}}]}\n\n'
            'data: [DONE]\n'
        )

        normalized = normalize_llm_json_text(text)

        self.assertEqual(normalized, '{"action":[{"done":{}}]}')

    def test_normalize_llm_json_text_rejects_metadata_only_sse(self) -> None:
        text = (
            'data: {"id":"","object":"chat.completion.chunk","choices":[],"usage":{"total_tokens":12}}\n\n'
            'data: [DONE]\n'
        )

        with self.assertRaisesRegex(ValueError, "no assistant content"):
            normalize_llm_json_text(text)

    def test_rewrite_google_images_navigation_actions_uses_direct_url_for_explicit_query(self) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [116, 116]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task='Open Safari, go to Google Images, search for "晴空塔" images, and stop once the image results page is visible.',
            next_goal="In Safari, click the Google Images tab and stop once thumbnails are visible.",
        )

        dumped = [action.model_dump(exclude_unset=True) for action in rewritten]
        self.assertEqual(dumped[1], {"wait": {}})
        self.assertIn("run_apple_script", dumped[0])
        self.assertIn(
            "https://www.google.com/search?tbm=isch&q=%E6%99%B4%E7%A9%BA%E5%A1%94",
            dumped[0]["run_apple_script"]["script"],
        )

    def test_rewrite_google_images_navigation_actions_uses_chrome_script_for_explicit_query(self) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [116, 116]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task='Open Google Chrome, go to Google Images, search for "晴空塔" images, and stop once the image results page is visible.',
            next_goal="In Google Chrome, click the Google Images tab and stop once thumbnails are visible.",
        )

        dumped = [action.model_dump(exclude_unset=True) for action in rewritten]
        self.assertIn('tell application "Google Chrome"', dumped[0]["run_apple_script"]["script"])
        self.assertIn(
            'set URL of active tab of front window to "https://www.google.com/search?tbm=isch&q=%E6%99%B4%E7%A9%BA%E5%A1%94"',
            dumped[0]["run_apple_script"]["script"],
        )

    def test_rewrite_google_images_navigation_actions_can_derive_current_query(self) -> None:
        actions = [
            ActionModel.model_validate({"Hotkey": {"key": "ESC"}}),
            ActionModel.model_validate({"Click": {"position": [131, 115]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="In Safari on the current Google search results page, switch to the Google Images results page and stop only when image thumbnails are visible.",
            next_goal="In Safari, dismiss the search suggestions if needed, then switch the current query to Google Images and stop once thumbnails are visible.",
        )

        dumped = [action.model_dump(exclude_unset=True) for action in rewritten]
        self.assertIn("run_apple_script", dumped[0])
        self.assertIn("new URL(window.location.href).searchParams.get('q')", dumped[0]["run_apple_script"]["script"])
        self.assertEqual(dumped[1], {"wait": {}})

    def test_rewrite_google_images_navigation_actions_can_derive_current_query_in_chrome(self) -> None:
        actions = [
            ActionModel.model_validate({"Hotkey": {"key": "ESC"}}),
            ActionModel.model_validate({"Click": {"position": [131, 115]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="In Google Chrome on the current Google search results page, switch to the Google Images results page and stop only when image thumbnails are visible.",
            next_goal="In Google Chrome, dismiss the search suggestions if needed, then switch the current query to Google Images and stop once thumbnails are visible.",
        )

        dumped = [action.model_dump(exclude_unset=True) for action in rewritten]
        self.assertIn('tell application "Google Chrome"', dumped[0]["run_apple_script"]["script"])
        self.assertIn("execute active tab of front window javascript", dumped[0]["run_apple_script"]["script"])
        self.assertIn("new URL(window.location.href).searchParams.get('q')", dumped[0]["run_apple_script"]["script"])

    def test_rewrite_google_images_navigation_actions_skips_non_image_tasks(self) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [233, 106]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="Open Safari and search the web for the Tokyo Skytree official website.",
            next_goal="Click the top search result and stop when the official site is visible.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_rewrite_google_images_navigation_actions_skips_non_image_tasks_even_if_goal_mentions_images(self) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [159, 108]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="Open Safari, search the web for the Tokyo Skytree official website, and stop on the Google web results page.",
            next_goal="In Safari, click the 「すべて」 tab to switch from Google Images to the standard Google web results page.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_rewrite_google_images_navigation_actions_replaces_done_only_when_goal_is_empty(self) -> None:
        actions = [ActionModel.model_validate({"done": {}})]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="In Safari on the current Google search results page for 晴空塔, switch to the Google Images results page and stop only when image thumbnails are visible.",
            next_goal="",
        )

        dumped = [action.model_dump(exclude_unset=True) for action in rewritten]
        self.assertIn("run_apple_script", dumped[0])
        self.assertEqual(dumped[1], {"wait": {}})

    def test_build_google_images_fallback_actions_uses_task_when_actor_response_is_empty(self) -> None:
        fallback = build_google_images_fallback_actions(
            task='Open Safari, go to Google Images, search for "Tokyo Skytree 晴空塔" images, and stop once the image results page is visible.',
            next_goal="",
        )

        self.assertIsNotNone(fallback)
        dumped = [action.model_dump(exclude_unset=True) for action in fallback]
        self.assertIn("run_apple_script", dumped[0])
        self.assertIn("Tokyo%20Skytree%20%E6%99%B4%E7%A9%BA%E5%A1%94", dumped[0]["run_apple_script"]["script"])
        self.assertEqual(dumped[1], {"wait": {}})

    def test_build_google_images_fallback_actions_prefers_chrome_when_task_names_it(self) -> None:
        fallback = build_google_images_fallback_actions(
            task='Open Google Chrome, go to Google Images, search for "Tokyo Skytree 晴空塔" images, and stop once the image results page is visible.',
            next_goal="",
        )

        self.assertIsNotNone(fallback)
        dumped = [action.model_dump(exclude_unset=True) for action in fallback]
        self.assertIn('tell application "Google Chrome"', dumped[0]["run_apple_script"]["script"])
        self.assertIn("Tokyo%20Skytree%20%E6%99%B4%E7%A9%BA%E5%A1%94", dumped[0]["run_apple_script"]["script"])

    def test_build_google_images_fallback_actions_returns_done_for_stop_goal(self) -> None:
        fallback = build_google_images_fallback_actions(
            task="In Safari on the current Google search results page for 晴空塔, switch to the Google Images results page and stop only when image thumbnails are visible.",
            next_goal="Stop. The Google Images results page for 晴空塔 is open in Safari and image thumbnails are visible.",
        )

        self.assertIsNotNone(fallback)
        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in fallback],
            [{"done": {}}],
        )

    def test_screenshot_to_dataurl_uses_resized_jpeg(self) -> None:
        img = Image.new("RGB", (1600, 900))
        pixels = [
            ((x * 13) % 256, (y * 7) % 256, ((x + y) * 5) % 256)
            for y in range(img.height)
            for x in range(img.width)
        ]
        img.putdata(pixels)

        data_url = screenshot_to_dataurl(img)

        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))

        payload = data_url.split(",", 1)[1]
        decoded = Image.open(io.BytesIO(base64.b64decode(payload)))
        self.assertLessEqual(decoded.width, 1024)


if __name__ == "__main__":
    unittest.main()
