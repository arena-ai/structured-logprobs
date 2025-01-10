![structured-logprobs](images/logo.png)

# structured-logprobs

This Python library is designed to enhance OpenAI chat completion responses by adding detailed information about token log probabilities.
This library works with OpenAI [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), which is a feature that ensures the model will always generate responses that adhere to your supplied JSON Schema, so you don't need to worry about the model omitting a required key, or hallucinating an invalid enum value.
It provides utilities to analyze and incorporate token-level log probabilities into structured outputs, helping developers understand the reliability of structured data extracted from OpenAI models.

## Purpose

![structured-logprobs](images/pitch.png)

The primary goal of `structured-logprobs` is to provide insights into the **reliability** of extracted data. By analyzing token-level log probabilities, the library enables:

- Understand how likely each token is based on the model's predictions.
- Detect low-confidence areas in responses for further review.

## Prerequisites

Before using this library, one should be familiar with:

- the OpenAI API and its client.
- the concept of log probabilities, a measure of the likelihood assigned to each token by the model.

## Key Features

The module contains a function for mapping characters to token indices (`map_characters_to_token_indices`) and two methods for incorporating log probabilities:

1. Adding log probabilities as a separate field in the response (`add_logprobs`).
2. Embedding log probabilities inline within the message content (`add_logprobs_inline`).

## Example

To use this library, first create a chat completion response with the OpenAI Python SDK, then enhance the response with log probabilities.
Here is an example of how to do that:

```python
from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from structured_logprobs import add_logprobs, add_logprobs_inline

# Initialize the OpenAI client
client = OpenAI(api_key="your-api-key")

schema_path = "path-to-your-json-schema"
with open(schema_path) as f:
        schema_content = json.load(f)

# Validate the schema content
response_schema = ResponseFormatJSONSchema.model_validate(schema_content)

# Create a chat completion request
completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages = [
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
    response_format=response_schema.model_dump(by_alias=True),
)

chat_completion = add_logprobs(completion)
chat_completion_inline = add_logprobs_inline(completion)
print(chat_completion.log_probs[0])
{'capital_of_France': -5.5122365e-07, 'the_two_nicest_colors': [-0.0033997903, -0.011364183612649998], 'die_shows': -0.48048785}
print(chat_completion_inline.choices[0].message.content)
{"capital_of_France": "Paris", "capital_of_France_logprob": -6.704273e-07, "the_two_nicest_colors": ["blue", "green"], "die_shows": 5.0, "die_shows_logprob": -2.3782086}
```

## Example JSON Schema

The `response_format` in the request body is an object specifying the format that the model must output. Setting to { "type": "json_schema", "json_schema": {...} } ensures the model will match your supplied JSON schema.

Below is the example of the JSON file that defines the schema used for validating the responses.

```python
{
    "type": "json_schema",
    "json_schema": {
        "name": "answears",
        "description": "Response to questions in JSON format",
        "schema": {
            "type": "object",
            "properties": {
                "capital_of_France": { "type": "string" },
                "the_two_nicest_colors": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["red", "blue", "green", "yellow", "purple"]
                    }
                },
                "die_shows": { "type": "number" }
            },
            "required": ["capital_of_France", "the_two_nicest_colors", "die_shows"],
            "additionalProperties": false
        },
        "strict": true
    }
}
```
