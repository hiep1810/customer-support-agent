import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GEMINI_API_KEY", "test-api-key")
os.environ.setdefault("GEMINI_MODEL", "test-model")

import backend.customer_support as customer_support


def test_generate_response(monkeypatch):
    """generate_response returns model output using provided template."""

    def fake_from_template(template: str):
        class DummyPrompt:
            def __or__(self, other):
                class DummyChain:
                    def invoke(self, inputs):
                        class Result:
                            content = f"{template} -> {inputs['query']}"
                        return Result()
                return DummyChain()
        return DummyPrompt()

    class DummyModel:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(customer_support.ChatPromptTemplate, "from_template", fake_from_template)
    monkeypatch.setattr(customer_support, "ChatGoogleGenerativeAI", DummyModel)

    state = {"query": "How do I reset my password?"}
    result = customer_support.generate_response(state, "Answer: {query}")

    expected = (
        "Answer: {query}\n\nRespond in the user's language and keep it concise."
        " -> How do I reset my password?"
    )
    assert result == {"response": expected}
