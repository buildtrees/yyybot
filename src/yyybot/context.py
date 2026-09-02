"""Conversation context assembly, independent from agent execution."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import Message


class ConversationContext:
    """Build model input messages and retain committed conversation history.

    ``build`` is intentionally side-effect free. Callers decide when a turn has
    completed successfully and persist it with ``record``. This leaves room for
    future memory, summarization, and retrieval policies without coupling them
    to the agent's model/tool execution loop.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = "You are a helpful personal assistant.",
        history: Sequence[Message] = (),
    ) -> None:
        self.system_prompt = system_prompt
        self._history = list(history)

    @property
    def history(self) -> tuple[Message, ...]:
        """Return an immutable snapshot of the committed history."""

        return tuple(self._history)

    def build(self, *incoming: Message) -> tuple[Message, ...]:
        """Assemble committed history and the messages for the current turn."""

        messages = [*self._history, *incoming]
        if self.system_prompt and not any(
            message.role == "system" for message in messages
        ):
            messages.insert(0, Message(role="system", content=self.system_prompt))
        return tuple(messages)

    def record(self, *messages: Message) -> None:
        """Commit completed user/assistant messages to conversation history."""

        self._history.extend(messages)

    def clear(self) -> None:
        """Clear committed history while retaining the configured system prompt."""

        self._history.clear()
