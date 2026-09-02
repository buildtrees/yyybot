from yyybot import ConversationContext, Message


def test_context_builds_system_history_and_current_turn_without_committing():
    previous_user = Message(role="user", content="Earlier")
    previous_assistant = Message(role="assistant", content="Answer")
    current_user = Message(role="user", content="Now")
    context = ConversationContext(history=(previous_user, previous_assistant))

    messages = context.build(current_user)

    assert messages == (
        Message(role="system", content="You are a helpful personal assistant."),
        previous_user,
        previous_assistant,
        current_user,
    )
    assert context.history == (previous_user, previous_assistant)


def test_context_records_completed_turn_and_respects_existing_system_message():
    system = Message(role="system", content="Custom")
    user = Message(role="user", content="Question")
    assistant = Message(role="assistant", content="Response")
    context = ConversationContext(system_prompt="Default", history=(system,))

    context.record(user, assistant)

    assert context.build() == (system, user, assistant)


def test_context_clear_preserves_system_prompt_configuration():
    context = ConversationContext(system_prompt="Stay concise")
    context.record(Message(role="user", content="Question"))

    context.clear()

    assert context.history == ()
    assert context.build() == (Message(role="system", content="Stay concise"),)
