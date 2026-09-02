from __future__ import annotations

import json

import pytest

from yyybot import (
    AgentResult,
    ConversationContext,
    Message,
    ModelResponse,
    SessionError,
    SessionManager,
    SessionNotFoundError,
    ToolCall,
)


def make_result(
    input_messages: tuple[Message, ...],
    *,
    answer: str = "The answer is 5.",
) -> AgentResult:
    tool_request = Message(
        role="assistant",
        tool_calls=(ToolCall("call-1", "add", {"a": 2, "b": 3}),),
    )
    tool_result = Message(
        role="tool", content='{"ok":true,"result":5}', tool_call_id="call-1"
    )
    final = Message(role="assistant", content=answer)
    responses = (
        ModelResponse(tool_request, "tool_calls", {"total_tokens": 12}),
        ModelResponse(final, "stop", {"total_tokens": 20}),
    )
    return AgentResult(
        final_message=final,
        messages=(*input_messages, tool_request, tool_result, final),
        responses=responses,
    )


def test_session_round_trips_complete_agent_turn_without_repeating_context(tmp_path):
    manager = SessionManager(tmp_path)
    created = manager.create(
        session_id="trip-planning",
        title="旅行计划",
        system_prompt="回答简洁",
    )
    user = Message(role="user", content="计算 2 + 3")
    context = manager.load_context(created.session_id)
    input_messages = context.build(user)
    result = make_result(input_messages)

    turn = manager.append_turn(
        created.session_id,
        incoming=(user,),
        result=result,
    )
    loaded = manager.load(created.session_id)

    assert loaded.title == "旅行计划"
    assert loaded.system_prompt == "回答简洁"
    assert loaded.turns == (turn,)
    assert loaded.messages == (user, *turn.generated)
    assert loaded.turns[0].responses == result.responses
    assert manager.load_context(created.session_id).build() == (
        Message(role="system", content="回答简洁"),
        *loaded.messages,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trip-planning.jsonl").read_text().splitlines()
    ]
    assert [record["type"] for record in records] == ["session", "turn"]
    assert records[1]["incoming"] == [{"role": "user", "content": "计算 2 + 3"}]
    assert len(records[1]["generated"]) == 3
    assert len(records[1]["responses"]) == 2


def test_second_turn_loads_first_turn_and_appends_only_new_messages(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.create(session_id="two-turns")

    first_user = Message(role="user", content="First")
    first_input = manager.load_context(session.session_id).build(first_user)
    manager.append_turn(
        session.session_id,
        incoming=(first_user,),
        result=make_result(first_input, answer="First answer"),
    )

    second_user = Message(role="user", content="Second")
    second_input = manager.load_context(session.session_id).build(second_user)
    manager.append_turn(
        session.session_id,
        incoming=(second_user,),
        result=make_result(second_input, answer="Second answer"),
    )

    loaded = manager.load(session.session_id)
    assert len(loaded.turns) == 2
    assert loaded.messages.count(first_user) == 1
    assert loaded.messages.count(second_user) == 1
    assert len((tmp_path / "two-turns.jsonl").read_text().splitlines()) == 3


def test_append_rejects_a_result_from_another_context(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.create(session_id="expected")
    incoming = Message(role="user", content="Current")
    wrong_input = ConversationContext().build(Message(role="user", content="Different"))

    with pytest.raises(SessionError, match="does not match"):
        manager.append_turn(
            session.session_id,
            incoming=(incoming,),
            result=make_result(wrong_input),
        )


def test_session_ids_cannot_escape_the_storage_directory(tmp_path):
    manager = SessionManager(tmp_path)

    with pytest.raises(ValueError, match="session_id"):
        manager.create(session_id="../outside")


def test_sessions_can_be_listed_and_deleted(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.create(session_id="temporary", title="Temporary")

    assert manager.list() == (session,)

    manager.delete(session.session_id)

    assert manager.list() == ()
    with pytest.raises(SessionNotFoundError):
        manager.load(session.session_id)


def test_session_round_trips_reasoning_content(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.create(session_id="reasoning")
    user = Message(role="user", content="Question")
    input_messages = manager.load_context(session.session_id).build(user)
    final = Message(
        role="assistant",
        content="Answer",
        reasoning_content="Reasoning summary",
        reasoning_signature="signature",
    )
    result = AgentResult(
        final_message=final,
        messages=(*input_messages, final),
        responses=(ModelResponse(final),),
    )

    manager.append_turn(session.session_id, incoming=(user,), result=result)
    loaded = manager.load(session.session_id)

    assert loaded.messages[-1].reasoning_content == "Reasoning summary"
    assert loaded.messages[-1].reasoning_signature == "signature"
