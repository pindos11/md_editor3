from __future__ import annotations

import difflib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib import request
from urllib.error import HTTPError, URLError


AI_BLOCK_RE = re.compile(
    r"<!--\s*AI:\s*(?!end\b)(?P<instruction>.*?)\s*-->(?P<body>.*?)<!--\s*AI:\s*end\s*-->",
    re.IGNORECASE | re.DOTALL,
)


class CompletionBackend(Protocol):
    def complete(self, prompt: str) -> str:
        pass


def is_llama_cpp_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


@dataclass(frozen=True)
class AICleanupBlock:
    start: int
    end: int
    instruction: str
    original: str


@dataclass(frozen=True)
class AICleanupResult:
    new_text: str
    diff: str
    block_count: int


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        max_tokens: int = 350,
        temperature: float = 0.1,
        repeat_penalty: float = 1.15,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.0,
        seed: int | None = 42,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        disable_thinking: bool = True,
        n_batch: int = 128,
        n_ubatch: int = 128,
        offload_kqv: bool = True,
        flash_attn: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise RuntimeError("llama-cpp-python is not installed. Install the llm extra first.") from exc
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.disable_thinking = disable_thinking
        kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "offload_kqv": offload_kqv,
            "flash_attn": flash_attn,
            "verbose": False,
        }
        if n_threads is not None and n_threads > 0:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)

    def complete(self, prompt: str) -> str:
        self.reset()
        no_think_stop = [
            "<!-- AI:",
            "CONTEXT_BEFORE:",
            "CONTEXT_AFTER:",
            "TARGET_TEXT:",
            "REPLACEMENT:",
            "TARGET_TEXT_END",
        ]
        thinking_stop = ["<!-- AI:", "<|im_end|>"]
        if self.disable_thinking and self._supports_template_thinking_toggle():
            raw_prompt = self._format_no_think_chat_prompt(prompt)
            response = self.llm(
                raw_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repeat_penalty=self.repeat_penalty,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed,
                stop=["<|im_end|>", *no_think_stop],
            )
            return str(response["choices"][0]["text"])
        if hasattr(self.llm, "create_chat_completion"):
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise Markdown editing function. Do not include reasoning or <think> blocks. Return only replacement text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repeat_penalty=self.repeat_penalty,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed,
                stop=thinking_stop if not self.disable_thinking else no_think_stop,
            )
            return str(response["choices"][0]["message"]["content"])
        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repeat_penalty=self.repeat_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=thinking_stop if not self.disable_thinking else no_think_stop,
        )
        return str(response["choices"][0]["text"])

    def reset(self) -> None:
        reset = getattr(self.llm, "reset", None)
        if callable(reset):
            reset()

    def _supports_template_thinking_toggle(self) -> bool:
        metadata = getattr(self.llm, "metadata", {}) or {}
        template = str(metadata.get("tokenizer.chat_template", ""))
        return "enable_thinking" in template

    @staticmethod
    def _format_no_think_chat_prompt(prompt: str) -> str:
        return (
            "<s><|im_start|>system\n"
            "You are a precise Markdown editing function. Output only the replacement Markdown. "
            "No reasoning, no notes, no repeated text."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout_seconds: int = 120,
        transport: Callable[[request.Request, int], bytes] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _default_transport

    def complete(self, prompt: str) -> str:
        if not self.base_url:
            raise RuntimeError("OpenAI-compatible base URL is not configured.")
        if not self.model:
            raise RuntimeError("OpenAI-compatible model name is not configured.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You rewrite marked Markdown blocks. Return only the replacement text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop": ["<<<END_REPLACEMENT>>>", "<!-- AI:", "Context after:", "Context before:"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            raw = self.transport(req, self.timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI-compatible API request failed: {exc.reason}") from exc
        data = json.loads(raw.decode("utf-8"))
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible API response did not contain choices[0].message.content.") from exc


class AnthropicMessagesBackend:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_seconds: int = 120,
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        transport: Callable[[request.Request, int], bytes] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.anthropic_version = anthropic_version
        self.transport = transport or _default_transport

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Anthropic API key is not configured.")
        if not self.model:
            raise RuntimeError("Anthropic model name is not configured.")
        payload = {
            "model": self.model,
            "system": "You rewrite marked Markdown blocks. Return only the replacement text.",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop_sequences": ["<<<END_REPLACEMENT>>>", "<!-- AI:", "Context after:", "Context before:"],
        }
        req = request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )
        try:
            raw = self.transport(req, self.timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic Messages API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Anthropic Messages API request failed: {exc.reason}") from exc
        data = json.loads(raw.decode("utf-8"))
        try:
            parts = data["content"]
            return "".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Anthropic Messages API response did not contain text content.") from exc


def _default_transport(req: request.Request, timeout_seconds: int) -> bytes:
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return response.read()


def find_ai_cleanup_blocks(markdown_text: str) -> list[AICleanupBlock]:
    blocks: list[AICleanupBlock] = []
    for match in AI_BLOCK_RE.finditer(markdown_text):
        instruction = match.group("instruction").strip()
        original = match.group("body").strip("\r\n")
        start = match.start()
        end = match.end()
        if not original.strip():
            next_start, next_end, next_text = _find_next_markdown_block(markdown_text, match.end())
            if next_text.strip():
                start = next_start
                end = next_end
                original = next_text.strip("\r\n")
        blocks.append(
            AICleanupBlock(
                start=start,
                end=end,
                instruction=instruction,
                original=original,
            )
        )
    return blocks


def _find_next_markdown_block(markdown_text: str, offset: int) -> tuple[int, int, str]:
    pos = offset
    while pos < len(markdown_text) and markdown_text[pos] in " \t\r\n":
        pos += 1
    if pos >= len(markdown_text):
        return offset, offset, ""

    lines = markdown_text[pos:].splitlines(keepends=True)
    taken: list[str] = []
    consumed = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if taken and not in_fence and not stripped:
            break
        if taken and not in_fence and stripped.startswith("<!-- AI:"):
            break
        taken.append(line)
        consumed += len(line)

    return pos, pos + consumed, "".join(taken)


def run_ai_cleanup(markdown_text: str, backend: CompletionBackend, context_chars: int = 0) -> AICleanupResult:
    blocks = find_ai_cleanup_blocks(markdown_text)
    replacements: list[tuple[AICleanupBlock, str]] = []
    for block in blocks:
        prompt = build_cleanup_prompt(markdown_text, block, context_chars=context_chars)
        raw_response = backend.complete(prompt)
        revised = normalize_model_response(raw_response)
        if not revised:
            if has_unclosed_think_block(raw_response):
                raise RuntimeError(
                    "The model used the entire response budget inside an unfinished <think> block and produced "
                    "no final replacement. Increase max tokens, disable thinking, or use a stronger model/backend."
                )
            raise RuntimeError(
                "The model returned an empty or marker-only replacement. "
                "Try lower max tokens/temperature, a chat-tuned model, or the OpenAI-compatible backend."
            )
        replacements.append((block, revised))

    new_text = apply_cleanup_replacements(markdown_text, replacements)
    diff = make_unified_diff(markdown_text, new_text)
    return AICleanupResult(new_text=new_text, diff=diff, block_count=len(blocks))


def build_cleanup_prompt(markdown_text: str, block: AICleanupBlock, context_chars: int = 0) -> str:
    before = markdown_text[max(0, block.start - context_chars) : block.start].strip()
    after = markdown_text[block.end : block.end + context_chars].strip()
    return f"""Edit the TARGET_TEXT according to this task:
{block.instruction}

Return only the edited TARGET_TEXT. Do not include reasoning, comments, labels, context, notes, or code fences. If expanding a list, keep every original top-level item and add concise indented bullets under each item.

CONTEXT_BEFORE:
{before}
CONTEXT_BEFORE_END

TARGET_TEXT:
{block.original}
TARGET_TEXT_END

CONTEXT_AFTER:
{after}
CONTEXT_AFTER_END

Edited TARGET_TEXT only:
"""


def normalize_model_response(response: str) -> str:
    cleaned = response.strip()
    cleaned = strip_think_blocks(cleaned)
    cleaned = _strip_prompt_echo_lines(cleaned)
    cleaned = re.sub(r"^<REVISED_BLOCK>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*</REVISED_BLOCK>$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:<<<)?REPLACEMENT(?:>>>)?:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(
        r"<<<END_REPLACEMENT>>>|(?:<<<)?CONTEXT_(?:BEFORE|AFTER)(?:>>>)?:|TARGET_TEXT(?:_END)?|<!--\s*AI:",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    cleaned = re.sub(r"\s*<<<END_REPLACEMENT>>>$", "", cleaned, flags=re.IGNORECASE)
    cleaned = _drop_marker_only_lines(cleaned)
    fence_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        embedded_fence = re.search(r"```(?:markdown|md)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if embedded_fence:
            cleaned = embedded_fence.group(1).strip()
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    return cleaned


def _strip_prompt_echo_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and _is_prompt_echo_line(lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


def _is_prompt_echo_line(line: str) -> bool:
    normalized = line.strip().lower().lstrip("-*0123456789. ")
    return normalized in {
        "return only the replacement for the target text.",
        "return only the edited target_text.",
        "edited target_text only:",
        "do not include reasoning, comments, labels, context, notes, or code fences.",
    }


def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*<think\b[^>]*>.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def has_unclosed_think_block(text: str) -> bool:
    lowered = text.lower()
    return "<think" in lowered and "</think>" not in lowered


def _drop_marker_only_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and _is_marker_only_line(lines[0]):
        lines.pop(0)
    while lines and _is_marker_only_line(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _is_marker_only_line(line: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:<<<)?(?:REPLACEMENT|TARGET_TEXT|CONTEXT_BEFORE|CONTEXT_AFTER)(?:>>>)?:?\s*",
            line,
            flags=re.IGNORECASE,
        )
    )


def apply_cleanup_replacements(
    markdown_text: str,
    replacements: list[tuple[AICleanupBlock, str]],
) -> str:
    result = markdown_text
    for block, replacement in sorted(replacements, key=lambda item: item[0].start, reverse=True):
        result = result[: block.start] + replacement + result[block.end :]
    return result


def make_unified_diff(old_text: str, new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="current.md",
            tofile="ai-cleaned.md",
        )
    )
