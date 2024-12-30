import pytest
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from structured_logprobs.main import add_logprobs, add_logprobs_inline

# Your token should be in a .env file OPENAI_API_KEY="..."
load_dotenv()


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_simple_parsed_completion_with_openai():
    client = OpenAI()

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
    completion = add_logprobs(completion)
    event = completion.choices[0].message.parsed
    print(event)
    # Test if logprobs are there in the expected way (when they will be)
    assert event.name == "Science Fair"


def test_add_logprobs(simple_parsed_completion, json_output):
    completion = add_logprobs(simple_parsed_completion)
    if isinstance(completion.value, ParsedChatCompletion):
        event = completion.value.choices[0].message.parsed
        assert event.name == "Science Fair"
    assert completion.log_probs[0] == json_output


def test_add_logprobs_inline(simple_parsed_completion, json_output_inline):
    completion = add_logprobs_inline(simple_parsed_completion)
    if isinstance(completion, ParsedChatCompletion):
        event = completion.choices[0].message.parsed
        assert event.name == "Science Fair"
    # Test if logprobs are there in the expected way (when they will be)
    assert completion.choices[0].message.content == json_output_inline
