from pathlib import Path

import pytest

from hermes.llm import LLM

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM(LLM):
    """Replies from a function of (prompt, system), or from a list in order."""

    provider = "test"
    model = "scripted"

    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        if callable(self.replies):
            return self.replies(prompt, system)
        return self.replies.pop(0)


@pytest.fixture
def olympics():
    from hermes import KnowledgeGraph

    return KnowledgeGraph.from_files(FIXTURES / "olympics.ttl")
