import json
from pathlib import Path

import pytest
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    date: str | None
    participants: list[str]


@pytest.fixture
def simple_parsed_completion(pytestconfig) -> ParsedChatCompletion[CalendarEvent] | None:
    base_path = Path(pytestconfig.rootdir)  # Base directory where pytest was run
    with open(base_path / "tests" / "simple_parsed_completion.json") as f:
        return ParsedChatCompletion[CalendarEvent].model_validate_json(f.read())
    return None


@pytest.fixture
def json_output():
    return {"name": -0.0001889152953, "date": -0.09505325175929999, "participants": [0.0, -2.0560767000000003e-06]}


@pytest.fixture
def json_output_inline():
    return json.dumps({
        "name": "Science Fair",
        "name_logprob": -0.0001889152953,
        "date": "Friday",
        "date_logprob": -0.09505325175929999,
        "participants": ["Alice", "Bob"],
    })
