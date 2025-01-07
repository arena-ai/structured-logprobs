import json
import os
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    date: str | None
    participants: list[str]


@pytest.fixture
def chat_completion(pytestconfig) -> ChatCompletion:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base_path = Path(pytestconfig.rootdir)  # Base directory where pytest was run
    schema_path = base_path / "tests" / "resources" / "questions_json_schema.json"
    with open(schema_path) as f:
        schema_content = json.load(f)

    # Validate the schema content
    response_schema = ResponseFormatJSONSchema.model_validate(schema_content)

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "I have three questions. The first question is: What is the capital of France? "
                    "The second question is: Which are the two nicest colors? "
                    "The third question is: Can you roll a die and tell me which number comes up?"
                ),
            }
        ],
        logprobs=True,
        # Serialize using alias names to match OpenAI API's expected format.
        # This ensures that the field 'schema_' is serialized as 'schema' to meet the API's naming conventions.
        response_format=response_schema.model_dump(by_alias=True),
    )
    return completion


@pytest.fixture
def parsed_chat_completion() -> ParsedChatCompletion:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # A simple data model
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    # A request with structured output
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        logprobs=True,
        response_format=CalendarEvent,
    )
    return completion


@pytest.fixture
def simple_parsed_completion(pytestconfig) -> ParsedChatCompletion[CalendarEvent] | None:
    base_path = Path(pytestconfig.rootdir)  # Base directory where pytest was run
    with open(base_path / "tests" / "resources" / "simple_parsed_completion.json") as f:
        return ParsedChatCompletion[CalendarEvent].model_validate_json(f.read())
    return None


@pytest.fixture
def json_output() -> dict[str, Any]:
    return {"name": -0.0001889152953, "date": -0.09505325175929999, "participants": [0.0, -2.0560767000000003e-06]}


@pytest.fixture
def json_output_inline() -> str:
    return json.dumps({
        "name": "Science Fair",
        "name_logprob": -0.0001889152953,
        "date": "Friday",
        "date_logprob": -0.09505325175929999,
        "participants": ["Alice", "Bob"],
    })


class TokenLogprob:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


@pytest.fixture
def data_token() -> list[TokenLogprob]:
    return [
        TokenLogprob(token="{", logprob=-1.9365e-07),  # Token index 0
        TokenLogprob(token='"a"', logprob=-0.01117),  # Token index 1
        TokenLogprob(token=': "', logprob=-0.00279),  # Token index 2
        TokenLogprob(token="he", logprob=-1.1472e-06),  # Token index 3
        TokenLogprob(token='llo"', logprob=-0.00851),  # Token index 4
        TokenLogprob(token=', "', logprob=-0.00851),  # Token index 5
        TokenLogprob(token="b", logprob=-0.00851),  # Token index 6
        TokenLogprob(token='": ', logprob=-0.00851),  # Token index 7
        TokenLogprob(token="12", logprob=-0.00851),  # Token index 8
        TokenLogprob(token=', "', logprob=-1.265e-07),  # Token index 9
        TokenLogprob(token='c"', logprob=-0.00851),  # Token index 10
        TokenLogprob(token=': [{"', logprob=-0.00851),  # Token index 11
        TokenLogprob(token="d", logprob=-1.265e-07),  # Token index 12
        TokenLogprob(token='":', logprob=-0.00851),  # Token index 13
        TokenLogprob(token="42", logprob=-0.00851),  # Token index 14
        TokenLogprob(token="}, ", logprob=-1.265e-07),  # Token index 15
        TokenLogprob(token="11", logprob=-0.00851),  # Token index 16
        TokenLogprob(token="]}", logprob=-1.265e-07),  # Token index 17
    ]


@pytest.fixture
def token_indices() -> list[int]:
    return [
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        9,
        9,
        10,
        10,
        11,
        11,
        11,
        11,
        11,
        12,
        13,
        13,
        14,
        14,
        15,
        15,
        15,
        16,
        16,
        17,
        17,
    ]
