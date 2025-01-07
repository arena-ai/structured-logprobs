import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from structured_logprobs.main import add_logprobs, add_logprobs_inline, map_characters_to_token_indices

load_dotenv()


def test_map_characters_to_token_indices(data_token, token_indices):
    result = map_characters_to_token_indices(data_token)

    assert result == token_indices
    assert result.count(1) == len(data_token[1].token)


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_add_logprobs_with_openai(chat_completion):
    completion = add_logprobs(chat_completion)
    assert list(completion.log_probs[0].keys()) == ["capital_of_France", "the_two_nicest_colors", "die_shows"]
    assert isinstance(list(completion.log_probs[0].values())[0], float)
    assert isinstance(list(completion.log_probs[0].values())[1], list)
    assert isinstance(list(completion.log_probs[0].values())[1][0], float)
    assert isinstance(list(completion.log_probs[0].values())[2], float)


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_add_logprobs_inline_with_openai(chat_completion):
    completion_inline = add_logprobs_inline(chat_completion)
    message_content = json.loads(completion_inline.choices[0].message.content)
    assert list(message_content.keys()) == [
        "capital_of_France",
        "capital_of_France_logprob",
        "the_two_nicest_colors",
        "die_shows",
        "die_shows_logprob",
    ]
    assert json.loads(completion_inline.choices[0].message.content)["capital_of_France"] == "Paris"
    assert isinstance(list(message_content.values())[0], str)
    assert isinstance(list(message_content.values())[1], float)
    assert isinstance(list(message_content.values())[2], list)
    assert isinstance(list(message_content.values())[2][1], str)
    assert isinstance(list(message_content.values())[3], float)
    assert isinstance(list(message_content.values())[4], float)


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_generic_completion_with_openai(pytestconfig, json_output):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base_path = Path(pytestconfig.rootdir)  # Base directory where pytest was run
    schema_path = base_path / "tests" / "resources" / "simple_json_schema.json"
    with open(schema_path) as f:
        schema_content = json.load(f)

    # Validate the schema content
    response_schema = ResponseFormatJSONSchema.model_validate(schema_content)

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        logprobs=True,
        # Serialize using alias names to match OpenAI API's expected format.
        # This ensures that the field 'schema_' is serialized as 'schema' to meet the API's naming conventions.
        response_format=response_schema.model_dump(by_alias=True),
    )
    chat_completion = add_logprobs(completion)
    _ = add_logprobs_inline(completion)
    assert list(chat_completion.log_probs[0].keys()) == list(json_output.keys())


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_add_logprobs_parsed_completion_with_openai(parsed_chat_completion, json_output):
    completion = add_logprobs(parsed_chat_completion)
    event = completion.value.choices[0].message.parsed
    assert event.name == "Science Fair"
    assert list(completion.log_probs[0].keys()) == list(json_output.keys())
    assert type(list(completion.log_probs[0].values())[0]) is type(list(json_output.values())[0])
    assert type(list(completion.log_probs[0].values())[1]) is type(list(json_output.values())[1])
    assert type(list(completion.log_probs[0].values())[2]) is type(list(json_output.values())[2])
    assert type(list(completion.log_probs[0].values())[2][1]) is type(list(json_output.values())[2][1])


@pytest.mark.skip(reason="We do not want to automate this as no OPENAI_API_KEY is on github yet")
def test_add_logprobs_inline_parsed_completion_with_openai(parsed_chat_completion, json_output_inline):
    completion_inline = add_logprobs_inline(parsed_chat_completion)
    message_content = json.loads(completion_inline.choices[0].message.content)
    assert list(message_content.keys()) == list(json.loads(json_output_inline).keys())
    assert list(message_content.values())[0] == "Science Fair"
    assert isinstance(list(message_content.values())[1], float)
    assert list(message_content.values())[2] == "Friday"
    assert isinstance(list(message_content.values())[3], float)
    assert list(message_content.values())[4] == ["Alice", "Bob"]


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
    assert completion.choices[0].message.content == json_output_inline
