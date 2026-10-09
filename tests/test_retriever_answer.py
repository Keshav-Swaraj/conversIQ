"""Offline checks for grounded-answer prompt and fallback behavior."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import retriever


def test_answer_prompt_requests_conversational_page_only_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    def fake_chatgroq(**settings):
        assert settings["model"] == retriever.ANSWER_MODEL

        def respond(prompt_value):
            system, human = prompt_value.to_messages()
            seen["system"] = str(system.content)
            seen["human"] = str(human.content)
            return AIMessage(content="Frank overhears a plot and is killed (page 21).")

        return RunnableLambda(respond)

    monkeypatch.setattr(retriever, "ChatGroq", fake_chatgroq)
    answer = retriever.generate_answer(
        "What happens at the Riddle House?", "[Page 21]\nFrank was killed.", (21,)
    )

    assert answer == "Frank overhears a plot and is killed (page 21)."
    assert "short, natural" in seen["system"]
    assert "do not invent line-number citations" in seen["system"]
    assert "[Page 21]\nFrank was killed." in seen["human"]


def test_no_source_skips_answer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(**settings):
        raise AssertionError("No answer model call expected without source text")

    monkeypatch.setattr(retriever, "ChatGroq", unexpected)
    assert retriever.generate_answer("Question?", "", ()) == (
        "The document does not contain relevant information to answer this question."
    )
