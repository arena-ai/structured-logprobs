**structured-logprobs** is a Python library that enhances OpenAI's structured outputs by adding information about token log probabilities.

![structured-logprobs](images/pitch.png)

This library is designed to provide valuable insights into the **reliability of an LLM's structured output**.
It works with OpenAI [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), which is a feature that ensures the model will always generate responses that adhere to a supplied JSON Schema, eliminating concerns about missing required keys or hallucinating invalid values.

## Purpose

By analyzing token-level log probabilities, the library enables:

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
