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
    build_image_content_block,
    llm_supports_response_format,
    rewrite_browser_address_bar_actions,
    rewrite_google_images_navigation_actions,
    rewrite_search_result_navigation_actions,
    screenshot_to_dataurl,
    strip_premature_done_actions,
    to_structured,
)
from src.agent.prompts import ActorPrompt_turix, PlannerPrompt
from src.agent.views import AgentOutput
from src.controller.registry.views import ActionModel
from src.utils.llm_response import normalize_llm_json_text


class DummyOutput(BaseModel):
    value: str


_MAIN_SPEC = spec_from_file_location(
    "main_for_tests",
    Path(__file__).resolve().parents[1] / "examples" / "main.py",
)
_MAIN = module_from_spec(_MAIN_SPEC)
assert _MAIN_SPEC.loader is not None
_MAIN_SPEC.loader.exec_module(_MAIN)


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

    def test_build_openai_compatible_llm_disables_response_format_for_custom_proxy(
        self,
    ) -> None:
        llm = _MAIN.build_openai_compatible_llm(
            model_name="gpt-5.4",
            api_key="test-key",
            base_url="https://proxy.pieixan.icu/v1",
        )

        self.assertFalse(getattr(llm, "_turix_supports_response_format"))

    def test_build_openai_compatible_llm_keeps_response_format_for_openai_cloud(
        self,
    ) -> None:
        llm = _MAIN.build_openai_compatible_llm(
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
        llm = _MAIN.CompatChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        result = llm._create_chat_result('{"action":[{"done":{}}]}')

        self.assertEqual(
            result.generations[0].message.content, '{"action":[{"done":{}}]}'
        )

    def test_compat_chat_openai_combines_outputs_without_token_usage(self) -> None:
        llm = _MAIN.CompatChatOpenAI(
            model="gpt-5.4",
            openai_api_key="test-key",
            openai_api_base="https://proxy.pieixan.icu/v1",
        )

        combined = llm._combine_llm_outputs(
            [
                {},
                {
                    "token_usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                    }
                },
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

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in filtered],
            [{"done": {}}],
        )

    def test_strip_premature_done_actions_removes_done_when_other_actions_exist(
        self,
    ) -> None:
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
            "data: [DONE]\n"
        )

        normalized = normalize_llm_json_text(text)

        self.assertEqual(normalized, '{"action":[{"done":{}}]}')

    def test_normalize_llm_json_text_rejects_metadata_only_sse(self) -> None:
        text = (
            'data: {"id":"","object":"chat.completion.chunk","choices":[],"usage":{"total_tokens":12}}\n\n'
            "data: [DONE]\n"
        )

        with self.assertRaisesRegex(ValueError, "no assistant content"):
            normalize_llm_json_text(text)

    def test_rewrite_google_images_navigation_actions_keeps_visual_navigation_for_explicit_query(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [116, 116]}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task='Open Safari, go to Google Images, search for "晴空塔" images, and stop once the image results page is visible.',
            next_goal="In Safari, click the Google Images tab and stop once thumbnails are visible.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_rewrite_google_images_navigation_actions_keeps_visual_navigation_when_query_must_be_derived(
        self,
    ) -> None:
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

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_rewrite_google_images_navigation_actions_skips_non_image_tasks(
        self,
    ) -> None:
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

    def test_rewrite_google_images_navigation_actions_skips_non_image_tasks_even_if_goal_mentions_images(
        self,
    ) -> None:
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

    def test_rewrite_google_images_navigation_actions_keeps_done_only_steps_even_when_goal_is_empty(
        self,
    ) -> None:
        actions = [ActionModel.model_validate({"done": {}})]

        rewritten = rewrite_google_images_navigation_actions(
            actions,
            task="In Safari on the current Google search results page for 晴空塔, switch to the Google Images results page and stop only when image thumbnails are visible.",
            next_goal="",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [{"done": {}}],
        )

    def test_rewrite_browser_address_bar_actions_prefers_cmd_l_over_toolbar_clicks(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [281, 53]}}),
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Safari and search the web for 2026 Japanese TV dramas.",
            next_goal="In Safari, click directly inside the top combined address/search bar text once, then press Command+A, type exactly: 2026 日剧 评分, and press Enter.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_browser_address_bar_actions_inserts_cmd_l_when_goal_assumes_focus(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Safari and correct the query on the current Google results page.",
            next_goal="In Safari, with the address/search bar already focused, press Command+A to select all existing text, type exactly: 2026 日剧 评分, then press Enter to run the search.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_browser_address_bar_actions_skips_non_toolbar_search_boxes(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [411, 188]}}),
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "Narita Express"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Safari and search inside the on-page help center widget.",
            next_goal="Click the help center search box on the webpage, replace its contents with Narita Express, and press Enter.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_rewrite_browser_address_bar_actions_drops_result_clicks_until_new_query_loads(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 冬季日剧 评分"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [332, 291]}}),
            ActionModel.model_validate(
                {"scroll_down": {"position": [331, 523], "dx": 0, "dy": 13}}
            ),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Safari and search the web for 2026 winter Japanese TV drama ratings.",
            next_goal="In Safari, press Command+L to focus the address/search bar, type exactly: 2026 冬季日剧 评分, and press Enter. On the results page, click a credible rating result.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 冬季日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_browser_address_bar_actions_dedupes_command_synonyms(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "COMMAND", "key2": "L", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Google Chrome and search the web for Japanese TV drama ratings.",
            next_goal="In Google Chrome, use Command+L to focus the main address bar/omnibox. Type exactly: 2026 日剧 评分 and press Enter. Wait until the Google search results page is visible.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_browser_address_bar_actions_preserves_popup_recovery_click_and_escape(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [732, 94]}}),
            ActionModel.model_validate({"Hotkey": {"key": "ESC"}}),
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "COMMAND", "key2": "L", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分"}}),
            ActionModel.model_validate({"Hotkey": {"key": "ENTER"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Google Chrome and search the web for Japanese TV drama ratings.",
            next_goal="In Google Chrome, first click the right Chrome window to make it active. Then press Escape to close the small floating search popup if it is still open. Next press Command+L to focus the real main address bar/omnibox, type exactly: 2026 日剧 评分, and press Enter. Do not use the page find box.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"Click": {"position": [732, 94]}},
                {"Hotkey": {"key": "ESC"}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_browser_address_bar_actions_normalizes_trailing_newline_submit(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [759, 129]}}),
            ActionModel.model_validate({"Hotkey": {"key": "ESC"}}),
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "COMMAND", "key2": "L", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分\n"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_browser_address_bar_actions(
            actions,
            task="Open Google Chrome and search the web for Japanese TV drama ratings.",
            next_goal="In Google Chrome, click the RIGHT Chrome window once to activate it. Then press Escape to close the small floating search popup if it is still open. Next press Command+L to focus the real main address bar/omnibox, type exactly: 2026 日剧 评分, and press Enter. Do not use the page find box.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"Click": {"position": [759, 129]}},
                {"Hotkey": {"key": "ESC"}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}},
                {"multi_Hotkey": {"key1": "CMD", "key2": "A", "key3": None}},
                {"input_text": {"text": "2026 日剧 评分"}},
                {"Hotkey": {"key": "ENTER"}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_stops_after_first_result_click(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [172, 311]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [749, 536]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [754, 536]}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="Open Safari and search the web for 2026 Japanese TV drama ratings.",
            next_goal="In Safari, click the top Douban result titled '2026年冬季日剧'. After that page opens, find a 2026 drama with a strong numeric rating visible on the page, click that drama title to open its dedicated rating page, and stop only when the drama title and the numeric rating are both clearly visible on screen.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"Click": {"position": [172, 311]}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_keeps_pre_click_scroll_and_single_retry(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate(
                {"scroll_down": {"position": [512, 640], "dx": 0, "dy": 78}}
            ),
            ActionModel.model_validate({"Click": {"position": [214, 288]}}),
            ActionModel.model_validate({"Click": {"position": [633, 514]}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="On the current Google search results page in Safari, open the correct Douban result.",
            next_goal="Use minimal scrolling to re-locate the exact Douban result titled '2026年冬季日剧', click it, and after that page opens choose one visible high-rated 2026 drama entry to open its dedicated page.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"scroll_down": {"position": [512, 640], "dx": 0, "dy": 78}},
                {"Click": {"position": [214, 288]}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_trims_clicks_for_visible_result_open_page_goal(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate(
                {"scroll_down": {"position": [224, 824], "dx": 0, "dy": 18}}
            ),
            ActionModel.model_validate({"Click": {"position": [214, 288]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [633, 514]}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="Open Safari, search the web for 2026 Japanese TV drama ratings, and open the correct Douban result.",
            next_goal="In Safari, scroll down through the Google results page until you find a Douban result with the title exactly or very closely matching '2026年冬季日剧'. Do not click non-Douban results. Once that Douban result is visible, click it to open the page.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"scroll_down": {"position": [224, 824], "dx": 0, "dy": 18}},
                {"Click": {"position": [214, 288]}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_trims_clicks_for_after_opening_it_goal(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [119, 313]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate(
                {"scroll_down": {"position": [251, 569], "dx": 0, "dy": 22}}
            ),
            ActionModel.model_validate({"Click": {"position": [276, 421]}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="Open Safari, search the web for 2026 Japanese TV dramas, and stop only when one drama title and rating are visible.",
            next_goal="On the current Google results page, click a credible rating/list result that contains ratings for 2026 Japanese dramas, preferably the visible Douban result titled '2026年冬季日剧'. After opening it, look for one specific 2026 Japanese drama with a high numeric public rating, then click into that drama's dedicated rating/detail page if needed.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"Click": {"position": [119, 313]}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_drops_open_app_on_current_results_page(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"open_app": {"app_name": "Google Chrome"}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [157, 256]}}),
            ActionModel.model_validate({"wait": {}}),
            ActionModel.model_validate({"Click": {"position": [318, 309]}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="Open Google Chrome and find a credible rating page for a 2026 Japanese drama.",
            next_goal="On the current Google results page in Chrome, click the blue result title '2026年冬季日剧' from Douban. After the Douban page opens, locate the entry '丰臣兄弟！' with rating 8.3 and click that title to open its dedicated page.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [
                {"Click": {"position": [157, 256]}},
                {"wait": {}},
            ],
        )

    def test_rewrite_search_result_navigation_actions_skips_window_activation_before_address_bar_recovery(
        self,
    ) -> None:
        actions = [
            ActionModel.model_validate({"Click": {"position": [759, 129]}}),
            ActionModel.model_validate({"Hotkey": {"key": "ESC"}}),
            ActionModel.model_validate(
                {"multi_Hotkey": {"key1": "CMD", "key2": "L", "key3": None}}
            ),
            ActionModel.model_validate({"input_text": {"text": "2026 日剧 评分\n"}}),
            ActionModel.model_validate({"wait": {}}),
        ]

        rewritten = rewrite_search_result_navigation_actions(
            actions,
            task="Open Google Chrome. Search the web for 2026 Japanese TV dramas and stop only when a drama title and rating are visible.",
            next_goal="In Google Chrome, click the RIGHT Chrome window once to activate it. Then press Escape to close the small floating search popup if it is still open. Next press Command+L to focus the REAL main address bar at the top of the Chrome window, replacing the GitHub URL. Type exactly: 2026 日剧 评分 and press Enter. Wait until a normal Google search results page is visible.",
        )

        self.assertEqual(
            [action.model_dump(exclude_unset=True) for action in rewritten],
            [action.model_dump(exclude_unset=True) for action in actions],
        )

    def test_screenshot_to_dataurl_preserves_more_visual_detail(self) -> None:
        img = Image.new("RGB", (2400, 1350))
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
        self.assertLessEqual(decoded.width, 1728)
        self.assertGreater(decoded.width, 1024)

    def test_build_image_content_block_defaults_to_high_detail(self) -> None:
        img = Image.new("RGB", (1200, 800), color="white")

        block = build_image_content_block(img)

        self.assertEqual(block["type"], "image_url")
        self.assertEqual(block["image_url"]["detail"], "high")
        self.assertTrue(block["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_actor_prompt_guides_precise_result_clicks_and_avoids_ads(self) -> None:
        prompt = ActorPrompt_turix(action_descriptions="", max_actions_per_step=5)
        content = prompt.get_system_message().content

        self.assertIn("blue title text", content)
        self.assertIn("Avoid sponsored", content)
        self.assertIn("shopping", content.lower())
        self.assertIn("Command+L", content)
        self.assertIn("Do not click a result in the same step", content)
        self.assertIn("destination page opens", content)
        self.assertIn("card body", content)

    def test_planner_prompt_prefers_graphical_navigation_for_browser_tasks(
        self,
    ) -> None:
        prompt = PlannerPrompt(action_descriptions="", max_actions_per_step=5)
        content = prompt.get_system_message().content

        self.assertIn("Prefer graphical", content)
        self.assertNotIn("Prioritize AppleScript/terminal for speed", content)


if __name__ == "__main__":
    unittest.main()
