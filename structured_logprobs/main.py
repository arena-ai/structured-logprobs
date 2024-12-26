from typing import TypeVar

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

CC = TypeVar("CC", bound=ChatCompletion)


def add_logprobs(chat_completion_response: CC) -> CC:
    """Summary line.

    Extended description of function.

    Args:
        bar: Description of input argument.

    Returns:
        Description of return value
    """

    # TODO NG: add the useful code there
    if isinstance(chat_completion_response, ParsedChatCompletion):
        # Process the Parsed case
        pass
    else:
        # The default case
        pass

    return chat_completion_response


if __name__ == "__main__":  # pragma: no cover
    pass
