import json
from typing import Any, TypeVar

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import BaseModel

from helpers import extract_json_data, extract_json_data_inline

CC = TypeVar("CC", bound=ChatCompletion)
MISSING_LOGPROBS_MESSAGE = "The 'logprobs' field is missing"


class ChatCompletionWithLogProbs(BaseModel):
    value: ChatCompletion
    log_probs: list[Any]


def map_characters_to_token_indices(extracted_data_token: list[ChatCompletionTokenLogprob]) -> list[int]:
    """
    Maps each character in the JSON string output to its corresponding token index.

    Args:
    extracted_data_token : A list of `TokenLogprob` objects, where each object represents a token and its data (such as the logprobs)

    Returns:
    A list of integers where each position corresponds to a character in the concatenated JSON string,
    and the integer at each position is the index of the token responsible for generating that specific character in the JSON string.

    Example:
    --------
    Given `extracted_data_token = [TokenLogprob(token='{'), TokenLogprob(token='"key1"'), TokenLogprob(token=': '), TokenLogprob(token='"value1"'), TokenLogprob(token='}')]`
    the JSON output is : '{"key1": "value1"}' and the function will return the list [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]

    """

    json_output = "".join(token_data.token for token_data in extracted_data_token)

    token_indices = [-1] * len(json_output)
    current_char_pos = 0

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        for _ in range(len(token_text)):
            token_indices[current_char_pos] = token_idx
            current_char_pos += 1

    return token_indices


def add_logprobs(chat_completion_response: ChatCompletion) -> ChatCompletionWithLogProbs:
    """
    Adds logprobs in the chat completion response and returns a
    ChatCompletionWithLogProbs object.

    Args:
        ChatCompletion: The OpenAI chat completion response.

    Returns:
        ChatCompletionWithLogProbs: An object containing:
            - A copy of the chat completion response.
            - A `log_probs` field, structured like the `message.content` of the response,
              where values are replaced with their respective log-probabilities.
    Raises:
        AttributeError: If any 'choice' in the response does not contain 'logprobs'.
    """

    logprobs_data = []
    for choice in chat_completion_response.choices:
        # Check if the 'logprobs' field is present
        if hasattr(choice, "logprobs") and choice.logprobs is not None and choice.logprobs.content is not None:
            extracted_data = choice.message.content
            # json_string = extracted_data[
            #    extracted_data.find("{") : extracted_data.rfind("}") + 1
            # ]
            # json_string=json.loads(json_string)
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

            # json_string = extracted_data[
            #    extracted_data.find("{") : extracted_data.rfind("}") + 1
            # ]
            logprobs_list = choice.logprobs.content
            token_indices = map_characters_to_token_indices(logprobs_list) if logprobs_list else []
            json_dict = extract_json_data_inline(extracted_data, logprobs_list, token_indices) if extracted_data else {}
            choice.message.content = json.dumps(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    return chat_completion_response


if __name__ == "__main__":  # pragma: no cover
    pass
