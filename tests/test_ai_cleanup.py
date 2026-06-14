from __future__ import annotations

import json

import pytest

from md_editor.ai_cleanup import (
    AnthropicMessagesBackend,
    LlamaCppBackend,
    OpenAICompatBackend,
    apply_cleanup_replacements,
    build_cleanup_prompt,
    find_ai_cleanup_blocks,
    has_unclosed_think_block,
    normalize_model_response,
    run_ai_cleanup,
    strip_think_blocks,
)


class FakeBackend:
    def complete(self, prompt: str) -> str:
        assert "rewrite next paragraph" in prompt
        assert "TARGET_TEXT:" in prompt
        return "This paragraph is clearer."


def test_finds_ai_cleanup_blocks():
    text = """Before

<!-- AI: rewrite next paragraph -->
Old paragraph.
<!-- AI: end -->

After
"""

    blocks = find_ai_cleanup_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].instruction == "rewrite next paragraph"
    assert blocks[0].original == "Old paragraph."


def test_run_ai_cleanup_replaces_marked_region_and_builds_diff():
    text = """Before

<!-- AI: rewrite next paragraph -->
Old paragraph.
<!-- AI: end -->

After
"""

    result = run_ai_cleanup(text, FakeBackend())

    assert result.block_count == 1
    assert "<!-- AI:" not in result.new_text
    assert "This paragraph is clearer." in result.new_text
    assert "-Old paragraph." in result.diff
    assert "+This paragraph is clearer." in result.diff


def test_cleanup_prompt_omits_surrounding_context_by_default():
    text = """<!-- AI: rewrite -->
Target
<!-- AI: end -->

Unrelated SQL task should not be in the prompt.
"""
    block = find_ai_cleanup_blocks(text)[0]

    prompt = build_cleanup_prompt(text, block)

    assert "Target" in prompt
    assert "Unrelated SQL task" not in prompt


def test_cleanup_prompt_can_include_explicit_surrounding_context():
    text = """Before context

<!-- AI: rewrite -->
Target
<!-- AI: end -->

After context
"""
    block = find_ai_cleanup_blocks(text)[0]

    prompt = build_cleanup_prompt(text, block, context_chars=50)

    assert "Before context" in prompt
    assert "After context" in prompt


def test_marker_wrapped_first_ordered_item_targets_only_marked_item():
    text = """<!-- AI: expand this list -->
1. OOP
<!-- AI: end -->
2. SOLID
3. SQL

After list
"""

    block = find_ai_cleanup_blocks(text)[0]

    assert block.original == "1. OOP"
    assert "2. SOLID" not in block.original


def test_apply_cleanup_handles_multiple_blocks_without_offset_errors():
    text = "<!-- AI: one -->a<!-- AI: end -->\n<!-- AI: two -->b<!-- AI: end -->"
    blocks = find_ai_cleanup_blocks(text)

    new_text = apply_cleanup_replacements(text, [(blocks[0], "A"), (blocks[1], "B")])

    assert new_text == "A\nB"


def test_normalize_model_response_strips_optional_wrapping():
    assert normalize_model_response("```markdown\nText\n```") == "Text"
    assert normalize_model_response("<REVISED_BLOCK>\nText\n</REVISED_BLOCK>") == "Text"
    assert normalize_model_response("Text\n<<<END_REPLACEMENT>>>\nMore text") == "Text"
    assert normalize_model_response("Text\n<!-- AI: end -->\nMore text") == "Text"
    assert normalize_model_response("<<<REPLACEMENT\nText") == "Text"
    assert normalize_model_response("<<<REPLACEMENT\n<<<REPLACEMENT\n") == ""
    assert normalize_model_response("Text\n```") == "Text"


def test_think_blocks_are_removed_from_model_output():
    assert strip_think_blocks("<think>reasoning</think>\nFinal text") == "Final text"
    assert strip_think_blocks("<think>unfinished reasoning\nFinal text") == ""
    assert normalize_model_response("<think>reasoning</think>\nFinal text") == "Final text"
    assert has_unclosed_think_block("<think>unfinished")


def test_run_ai_cleanup_reports_unfinished_thinking():
    class ThinkingOnlyBackend:
        def complete(self, prompt: str) -> str:
            return "<think>unfinished reasoning"

    text = "<!-- AI: rewrite -->\nTarget\n<!-- AI: end -->"

    with pytest.raises(RuntimeError, match="unfinished <think>"):
        run_ai_cleanup(text, ThinkingOnlyBackend())


def test_openai_compat_backend_posts_chat_completion_request():
    captured = {}

    def fake_transport(req, timeout_seconds):
        captured["url"] = req.full_url
        captured["timeout"] = timeout_seconds
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "Rewritten text"}}]}).encode("utf-8")

    backend = OpenAICompatBackend(
        base_url="https://proxy.example/v1/",
        api_key="test-key",
        model="work-model",
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        transport=fake_transport,
    )

    assert backend.complete("Prompt") == "Rewritten text"
    assert captured["url"] == "https://proxy.example/v1/chat/completions"
    assert captured["timeout"] == 120
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "work-model"
    assert captured["payload"]["top_p"] == 0.8
    assert captured["payload"]["frequency_penalty"] == 0.1
    assert captured["payload"]["presence_penalty"] == 0.2
    assert captured["payload"]["messages"][1]["content"] == "Prompt"


def test_anthropic_messages_backend_posts_messages_request():
    captured = {}

    def fake_transport(req, timeout_seconds):
        captured["url"] = req.full_url
        captured["timeout"] = timeout_seconds
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return json.dumps({"content": [{"type": "text", "text": "Rewritten text"}]}).encode("utf-8")

    backend = AnthropicMessagesBackend(
        api_key="test-key",
        model="claude-test",
        top_p=0.7,
        timeout_seconds=90,
        transport=fake_transport,
    )

    assert backend.complete("Prompt") == "Rewritten text"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["timeout"] == 90
    assert captured["headers"]["X-api-key"] == "test-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-test"
    assert captured["payload"]["top_p"] == 0.7
    assert captured["payload"]["messages"][0]["content"] == "Prompt"


def test_llama_backend_thinking_toggle_controls_no_think_prompt_path():
    backend = object.__new__(LlamaCppBackend)
    backend.disable_thinking = True
    backend.llm = type("FakeLlama", (), {"metadata": {"tokenizer.chat_template": "{% if enable_thinking %}"}})()
    assert backend._supports_template_thinking_toggle()
    assert backend.disable_thinking is True

    backend.disable_thinking = False
    assert backend.disable_thinking is False


def test_llama_backend_reset_calls_underlying_model_reset():
    class FakeLlama:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    backend = object.__new__(LlamaCppBackend)
    backend.llm = FakeLlama()

    backend.reset()

    assert backend.llm.reset_called


def test_llama_backend_stores_generation_seed():
    backend = object.__new__(LlamaCppBackend)
    backend.seed = 42

    assert backend.seed == 42
