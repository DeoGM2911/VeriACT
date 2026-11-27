"""Provider-agnostic chat client. The loop only sees `ChatClient`, which has one method, `send`.

`OpenAICompatibleClient` is the one implementation: it talks to a local vLLM server or any
OpenAI-compatible API via the `openai` package. Another provider can be added by subclassing
`ChatClient` and extending `build_client`.
"""
import base64
import dataclasses
import json
from typing import Any, Optional

from openai import OpenAI


@dataclasses.dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclasses.dataclass
class ChatReply:
    text: str  # the model's own text output this turn
    tool_calls: list
    output_tokens: int
    total_tokens: int
    raw: Any  # provider response, kept for the trace log
    reasoning_content: str = ""  # thinking-mode output (e.g. vLLM's reasoning_content), if any
    raw_response_json: str = ""  # unparsed response body; only populated when cfg.trace.log_raw_response


class ChatClient:
    """One method: send the running conversation, get back text + tool calls."""

    def send(self, messages: list, tools: list) -> ChatReply:
        raise NotImplementedError


class TokenTrackingChatClient(ChatClient):
    """Wraps another `ChatClient`, accumulating prompt/completion tokens across every `send()`
    made through it -- retries and tool round-trips included, not just the final `play` reply
    (whose own usage is what `tracelog.py`'s `usage` field records).

    `prompt_tokens` is derived as `total_tokens - output_tokens` (OpenAI-style usage guarantees
    `total = prompt + completion`).

    `loop.py` calls `reset_turn()` at the start of each turn and logs `prompt_tokens` /
    `completion_tokens` with that turn's trace record, including on a protocol failure.
    """

    def __init__(self, inner: ChatClient):
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def send(self, messages: list, tools: list) -> ChatReply:
        reply = self._inner.send(messages, tools)
        self.completion_tokens += reply.output_tokens
        self.prompt_tokens += max(reply.total_tokens - reply.output_tokens, 0)
        return reply

    def reset_turn(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0


class OpenAICompatibleClient(ChatClient):
    def __init__(self, base_url: str, api_key: str, model: str, max_output_tokens: int, temperature: float,
                 log_raw_response: bool = False, top_p: Optional[float] = None,
                 reasoning_effort: Optional[str] = None, timeout: float = 1800.0,
                 max_retries: int = 10):
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                              timeout=timeout, max_retries=max_retries)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._log_raw_response = log_raw_response
        self._top_p = top_p
        self._reasoning_effort = reasoning_effort

    def send(self, messages: list, tools: list) -> ChatReply:
        kwargs = dict(
            model=self._model,
            messages=messages,
            tools=tools,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        # Both left out of kwargs entirely when unset, not passed as None -- some
        # OpenAI-compatible servers reject an explicit null for a field they don't expect.
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._reasoning_effort is not None:
            # Sent via chat_template_kwargs rather than the top-level `reasoning_effort` field,
            # which vLLM validates against a fixed enum (none/low/medium/high). The template
            # kwargs are passed unvalidated to the model's chat template, which may or may not
            # read this key.
            kwargs["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": self._reasoning_effort}}
        raw_response_json = ""
        if self._log_raw_response:
            # Keep the unparsed body so fields the openai SDK's schema doesn't declare are visible.
            http_resp = self._client.chat.completions.with_raw_response.create(**kwargs)
            raw_response_json = http_resp.text
            resp = http_resp.parse()
        else:
            resp = self._client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            for tc in (choice.message.tool_calls or [])
        ]
        usage = resp.usage
        return ChatReply(
            text=text,
            tool_calls=tool_calls,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            raw=resp,
            # The field name isn't standardized across reasoning parsers: most use
            # "reasoning_content", some (e.g. vLLM's gemma4 parser) use "reasoning". Check both.
            reasoning_content=(
                getattr(choice.message, "reasoning_content", None)
                or getattr(choice.message, "reasoning", None)
                or ""
            ),
            raw_response_json=raw_response_json,
        )


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def image_content(png_bytes: bytes) -> dict:
    """An OpenAI-style image content block from raw PNG bytes."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def build_client(cfg) -> ChatClient:
    if cfg.model.provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=cfg.model.base_url,
            api_key=cfg.api_key(),
            model=cfg.model.model,
            max_output_tokens=cfg.model.max_output_tokens,
            temperature=cfg.model.temperature,
            log_raw_response=cfg.trace.log_raw_response,
            top_p=cfg.model.top_p,
            reasoning_effort=cfg.model.reasoning_effort,
            timeout=cfg.model.request_timeout_seconds,
            max_retries=cfg.model.max_retries,
        )
    raise NotImplementedError(
        f"model.provider={cfg.model.provider!r} is not implemented (only openai_compatible is)."
    )
