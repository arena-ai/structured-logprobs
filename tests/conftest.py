from pathlib import Path

import pytest
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


@pytest.fixture
def simple_parsed_completion(pytestconfig) -> ParsedChatCompletion[CalendarEvent] | None:
    base_path = Path(pytestconfig.rootdir)  # Base directory where pytest was run
    with open(base_path / "tests" / "simple_parsed_completion.json") as f:
        return ParsedChatCompletion[CalendarEvent].model_validate_json(f.read())
    return None
