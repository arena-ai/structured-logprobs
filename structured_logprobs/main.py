import json
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import BaseModel

from structured_logprobs.helpers import extract_json_data, extract_json_data_inline

MISSING_LOGPROBS_MESSAGE = "The 'logprobs' field is missing"

"""

This module provides utilities to work with OpenAI chat completion responses,
enhancing them by embedding log probabilities into the data.
The module contains a function for mapping characters to token indices (`map_characters_to_token_indices`) and two methods for incorporating log probabilities:
1. Adding log probabilities as a separate field in the response (`add_logprobs`).
2. Embedding log probabilities inline within the message content (`add_logprobs_inline`).

Classes:
    - ChatCompletionWithLogProbs: Represents a chat completion response with added log probabilities.

"""


class ChatCompletionWithLogProbs(BaseModel):
    value: ChatCompletion
    log_probs: list[Any]


def map_characters_to_token_indices(extracted_data_token: list[ChatCompletionTokenLogprob]) -> list[int]:
    """
    Maps each character in the JSON string output to its corresponding token index.

    Args:
    extracted_data_token : A list of `TokenLogprob` objects, where each object represents a token and its associated data.

    Returns:
    A list of integers where each position corresponds to a character in the concatenated JSON string,
    and the integer at each position is the index of the token responsible for generating that specific character.
    Example:
        >>> tokens = [ChatCompletionTokenLogprob(token='{'),
                      ChatCompletionTokenLogprob(token='"key1"'),
                      ChatCompletionTokenLogprob(token=': '),
                      ChatCompletionTokenLogprob(token='"value1"'),
                      ChatCompletionTokenLogprob(token='}')]
        >>> map_characters_to_token_indices(tokens)
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    """

    token_indices = []

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        token_indices.extend([token_idx] * len(token_text))

    return token_indices


def add_logprobs(chat_completion_response: ChatCompletion) -> ChatCompletionWithLogProbs:
    """
    Adds log probabilities to the chat completion response and returns a
    ChatCompletionWithLogProbs object.

    Args:
        chat_completion_response: The OpenAI chat completion response.

    Returns:
        An object containing:
            - The original chat completion response.
            - A `log_probs` field, structured like the message.content of the response,
              where values are replaced with their respective log-probabilities.
    Raises:
        AttributeError: If any 'choice' in the response does not contain 'logprobs'.

    """

    logprobs_data = []
    for choice in chat_completion_response.choices:
        # Check if the 'logprobs' field is present
        if hasattr(choice, "logprobs") and choice.logprobs is not None and choice.logprobs.content is not None:
            extracted_data = choice.message.content
            logprobs_list = choice.logprobs.content
            token_indices = map_characters_to_token_indices(logprobs_list) if logprobs_list else []
            json_dict = extract_json_data(extracted_data, logprobs_list, token_indices) if extracted_data else {}
            logprobs_data.append(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    chat_completion_with_logprobs = ChatCompletionWithLogProbs(value=chat_completion_response, log_probs=logprobs_data)
    return chat_completion_with_logprobs


def add_logprobs_inline(chat_completion_response: ChatCompletion) -> ChatCompletion:
    """
    Embeds inline log probabilities into the content of the message in the chat completion response.

    Args:
        ChatCompletion: The OpenAI chat completion response.

    Returns:
        ChatCompletion: The modified chat completion response object, where the content of the message
            is replaced with a dictionary that includes also inline log probabilities for atomic values.

    Raises:
        AttributeError: If the 'logprobs' field is not present in the response.
    """

    for choice in chat_completion_response.choices:
        # Check if the 'logprobs' field is present
        if hasattr(choice, "logprobs") and choice.logprobs is not None and choice.logprobs.content is not None:
            extracted_data = choice.message.content
            logprobs_list = choice.logprobs.content
            token_indices = map_characters_to_token_indices(logprobs_list) if logprobs_list else []
            json_dict = extract_json_data_inline(extracted_data, logprobs_list, token_indices) if extracted_data else {}
            choice.message.content = json.dumps(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    return chat_completion_response


if __name__ == "__main__":  # pragma: no cover
    pass
